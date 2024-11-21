import os
import sys
import json

from decimal import Decimal
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from typing import Annotated, Any, Dict, List, Sequence, Tuple, TypedDict

import time
from datetime import datetime

import threading
from queue import Queue
import signal

import boto3
from botocore.config import Config
from batchfold.batchfold_environment import BatchFoldEnvironment
from batchfold.batchfold_target import BatchFoldTarget
from batchfold.jackhmmer_job import JackhmmerJob
from batchfold.alphafold2_job import AlphaFold2Job

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import graph, END, StateGraph
from langchain.prompts import ChatPromptTemplate

from Bio import SeqIO

# import local modules
dir_current = os.path.abspath("")
dir_parent = os.path.dirname(dir_current)
if dir_parent not in sys.path:
    sys.path.append(dir_parent)
    
class AgentState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    next_step: str | None
    files: Dict[str, Any]
    current_response: str | None
    proteins: BatchFoldTarget | None

# set region and model name
aws_region = "us-west-2"
model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
temperature = 0
max_tokens = 50000

# Set bedrock configs
bedrock_config = Config(
    connect_timeout=3000, read_timeout=3000, retries={"max_attempts": 10}
)

# Create a bedrock runtime client
bedrock_rt = boto3.client(
    "bedrock-runtime", region_name=aws_region, config=bedrock_config
)

# Create AWS clients
boto_session = boto3.session.Session()

batch_environment = BatchFoldEnvironment(boto_session=boto_session)

S3_BUCKET = batch_environment.default_bucket
print(f" S3 bucket name is {S3_BUCKET}")

# setting dynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Experiment')

# Tool list

# Agent using LangGraph
class DrugDiscoveryAssistantAgent:
    def __init__(self, **kwargs):
        self.llm = ChatBedrockConverse(
            client=bedrock_rt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.state = {
            "messages": [],
            "next_step": None,
            "files": {},
            "current_response": None,
        }
        self.TOOLs = [
            {
                "name": "alphafold_protein_structure_prediction",
                "function_name": self.alphafold_protein_structure_prediction,
                "description": "Predict protein structure using Multiple Sequence Alignment",
                "input_file": "fasta"
            }
        ]
        self.graph = self._create_agent_graph()

    def route(self, state: AgentState) -> Dict[str, str]:
        """
        사용자의 입력을 분석하여 적절한 다음 단계의 작업을 결정하는 라우팅 함수입니다.

        Parameters
        ----------
        state : AgentState
            현재 에이전트의 상태를 담고 있는 딕셔너리. 
            반드시 'messages' 키를 포함해야 하며, 이는 대화 히스토리를 담고 있음

        Returns
        -------
        Dict[str, str]
            업데이트된 상태 딕셔너리. 다음 중 하나의 'next_step' 값을 포함:
            - 'basic_qa': 일반적인 질문답변
            - 'data_analysis': 데이터 분석 작업
            - 'other_tool_use': 외부 도구 사용
            - 'terminate': 대화 종료

        Notes
        -----
        - 사용자가 "종료"를 입력하면 'terminate'를 반환
        - LLM을 사용하여 사용자의 입력을 분석하고 적절한 다음 단계를 결정
        - 입력된 상태 딕셔너리를 복사하여 새로운 상태를 반환

        Examples
        --------
        >>> state = {"messages": [Message(content="단백질 구조를 예측하고 싶습니다")]}
        >>> result = route(state)
        >>> print(result["next_step"])
        'other_tool_use'
        """
        messages = state["messages"]
        if messages[-1].content.lower() in ["종료"]:
            return {"next_step": "terminate"}

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 신약 개발 분야의 데이터 분석과 질문 답변을 돕는 AI 연구 어시스턴트입니다.
            사용자의 입력을 분석하여 다음 세 가지 작업 중 가장 적절한 것을 선택해야 합니다:

            1. "basic_qa" - 다음과 같은 경우 선택:
               - 신약 개발 관련 일반적인 지식 질문
               - 용어나 개념에 대한 설명 요청
               - 간단한 정보 조회나 설명이 필요한 경우

            2. "data_analysis" - 다음과 같은 경우 선택:
               - 데이터 파일(CSV 등)의 분석 요청
               - 통계적 분석이나 데이터 시각화 요청
               - 데이터셋 기반의 인사이트 도출 요청

            3. "other_tool_use" - 다음과 같은 경우 선택:
               - AlphaFold 등 외부 도구 사용 필요
               - 단백질 구조 예측 요청

            반드시 위 세 옵션 중 하나만 답변으로 제시하세요: "basic_qa", "data_analysis", "other_tool_use"
            """),
            ("human", "다음 사용자 메시지를 분석하여 적절한 작업을 선택하세요:\n{input}")
        ])

        response = self.llm.invoke(prompt.format_messages(input=messages[-1].content))
        action = response.content.strip()

        new_state = state.copy()
        new_state["next_step"] = action
        return new_state
    
    def handle_basic_qa(self, state: AgentState) -> AgentState:
        """
        신약 개발 관련 일반적인 질문에 대해 답변을 생성하는 함수입니다.

        Parameters
        ----------
        state : AgentState
            현재 에이전트의 상태를 담고 있는 딕셔너리.
            필수 키:
            - 'messages': 대화 히스토리를 담고 있는 리스트
            - 최신 메시지는 messages[-1]에서 접근 가능

        Returns
        -------
        AgentState
            업데이트된 상태 딕셔너리. 다음 키를 포함:
            - 'current_response': LLM이 생성한 답변 문자열

        Notes
        -----
        - LLM을 사용하여 사용자의 질문에 대한 답변을 생성
        - 신약 개발 도메인에 특화된 지식을 바탕으로 응답
        - 입력된 상태 딕셔너리를 복사하여 새로운 상태를 반환

        Examples
        --------
        >>> state = {"messages": [Message(content="단백질 키나아제란 무엇인가요?")]}
        >>> result = handle_basic_qa(state)
        >>> print(result["current_response"])
        '단백질 키나아제는 단백질의 특정 아미노산에 인산기를 전달하는 효소입니다...'
        """
        messages = state["messages"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 신약 개발 분야의 전문 지식을 갖춘 AI 연구 어시스턴트입니다.
            다음 영역에 대해 정확하고 체계적인 답변을 제공해야 합니다:

            전문 영역:
            - 약물 작용 메커니즘
            - 단백질 구조와 기능
            - 신약 개발 과정과 방법론
            - 질병 기전과 치료 타겟
            - 약물 동태학과 약력학
            - 임상 시험 단계와 규제

            답변 요구사항:
            1. 과학적 정확성을 최우선으로 함
            2. 필요한 경우 전문 용어를 쉽게 풀어서 설명
            3. 관련된 예시나 비유를 적절히 활용
            4. 답변의 한계나 불확실성이 있다면 분명하게 명시

            답변 구조:
            1. 핵심 개념 설명
            2. 상세 설명 및 예시
            3. 실제 적용 사례나 중요성
            4. 필요한 경우 추가 참고 사항"""),
            ("human", "{input}")
        ])

        response = self.llm.invoke(prompt.format_messages(input=messages[-1].content))

        new_state = state.copy()
        new_state["current_response"] = response.content
        return new_state

    def handle_data_analysis(self, state: AgentState) -> AgentState:
        """
        분자 데이터를 분석하여 특성을 예측하는 데이터 분석 함수입니다.

        Parameters
        ----------
        state : AgentState
            현재 에이전트의 상태를 담고 있는 딕셔너리.
            필수 키:
            - 'messages': 대화 히스토리
            - 'files': 분석에 필요한 파일들을 담고 있는 딕셔너리
              - 'training.csv': 학습 데이터
              - 'test.csv': 테스트 데이터

        Returns
        -------
        AgentState
            업데이트된 상태 딕셔너리. 다음 키를 포함:
            - 'current_response': LLM이 생성한 예측 결과

        Raises
        ------
        FileNotFoundError
            필요한 파일(training.csv 또는 test.csv)이 없는 경우

        Notes
        -----
        - 학습 데이터를 기반으로 분자의 특성을 학습
        - 테스트 데이터의 분자에 대해 특성을 예측
        - 예측 결과를 CSV 파일로 저장
        - 입력된 상태 딕셔너리를 복사하여 새로운 상태를 반환

        Examples
        --------
        >>> state = {
        ...     "messages": [Message(content="분자의 용해도를 예측해주세요")],
        ...     "files": {
        ...         "training.csv": "SMILES,solubility\\nC1=CC=CC=C1,2.3\\n",
        ...         "test.csv": "SMILES\\nC1=CC=CN=C1\\n"
        ...     }
        ... }
        >>> result = handle_data_analysis(state)
        >>> print("예측 결과가 저장되었습니다.")
        """
        messages = state["messages"]
        files = state["files"]

        required_files = {"training.csv", "test.csv"}
        missing_files = required_files - files.keys()

        if missing_files:
            raise FileNotFoundError(f"필요한 파일이 없습니다: {', '.join(missing_files)}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 신약 개발 분야의 데이터 분석 전문가입니다. 
            분자 데이터를 분석하여 특성을 예측하는 작업을 수행해야 합니다.

            작업 과정:
            1. 학습 데이터 분석
               - 분자 구조(SMILES)와 특성 간의 관계 파악
               - 구조-활성 관계(SAR) 패턴 학습
               - 특성값의 분포와 범위 확인

            2. 예측 수행
               - 학습된 패턴을 바탕으로 테스트 분자의 특성 예측
               - 유사 구조를 가진 분자들의 특성을 참조
               - 예측값이 합리적인 범위 내에 있는지 확인

            3. 출력 형식
               - CSV 형식으로 결과 제공
               - 컬럼: input(SMILES), output(예측값), property name(특성 이름)
               - 부가 설명이나 주석 없이 데이터만 학습데이터와 동일한 형태로 제공
               - 각 행은 하나의 분자에 대한 예측 결과만 포함

            주의사항:
            - 모든 테스트 분자에 대해 예측 제공
            - 결측값이나 오류가 없어야 함
            - SMILES 문자열의 정확한 처리
            - 특성값의 단위와 형식 유지"""),
            ("human", """Training data: {training_data}\n
            Test data: {test_data}\n
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            input = messages[-1].content,
            training_data = files['training.csv'],
            test_data = files['test.csv'],
            )
        )
    
        def _parse_claude_response(response: str) -> List[Dict]:
            """
            LLM의 응답에서 분자와 예측된 특성을 추출
            """
            # 여기에 응답 파싱 로직을 구현
            # 예시: 응답이 CSV 형식이라고 가정
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(response))
                return df.to_dict('records')
            except:
                # CSV 파싱이 실패할 경우의 대체 로직
                # 정규표현식이나 다른 파싱 방법을 사용할 수 있습니다
                pass

        def _save_predictions_to_csv(predictions: List[Dict], filename: str = "molecule/test_prediction.csv"):
            """예측 결과를 CSV 파일로 저장"""
            df = pd.DataFrame(predictions)
            df.to_csv(filename, index=False)
            print(f"예측 결과가 {filename}에 저장되었습니다.")

        print(response)
        predictions = _parse_claude_response(response.content)
        _save_predictions_to_csv(predictions)

        new_state = state.copy()
        new_state["current_response"] = response
        return new_state
    
    def tool_use(self, state: AgentState) -> AgentState:
        """
        사용자의 입력을 분석하여 적절한 외부 도구를 선택하는 라우팅 함수입니다.

        Parameters
        ----------
        state : AgentState
            현재 에이전트의 상태를 담고 있는 딕셔너리.
            필수 키:
            - 'messages': 대화 히스토리를 담고 있는 리스트
            - 최신 메시지는 messages[-1]에서 접근 가능

        Returns
        -------
        AgentState
            업데이트된 상태 딕셔너리. 다음 키를 포함:
            - 'next_step': 선택된 도구의 이름 문자열

        Notes
        -----
        - TOOLS에 정의된 도구들 중 하나를 선택
        - TOOLS의 각 항목은 다음 키를 포함하는 딕셔너리:
          - 'name': 도구의 이름
          - 'description': 도구의 기능 설명
        - 입력된 상태 딕셔너리를 복사하여 새로운 상태를 반환
        """  
        messages = state["messages"]

        tools_prompt = ""
        tools_types = ""
        tools_num = len(self.TOOLs)
        for index, tool in enumerate(self.TOOLs):
            tools_prompt += f'{index+1}. "{tool["name"]}" - {tool["description"]} '

            if index < tools_num-1:
                tools_types += f'"{tool["name"]}'
                tools_types += ', '
            else:
                tools_types += "or "
                tools_types += f'"{tool["name"]}'

        _nums = {1:'one',
            2:'two',
            3:'three',
            4:'four',
            5:'five',
            6:'six',
            7:'seven',
            8:'eight',
            9:'nine'}

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 신약 개발 분야에서 다양한 외부 도구를 활용하는 AI 어시스턴트입니다.
            사용자의 요청을 분석하여 가장 적절한 도구를 선택해야 합니다.

            사용 가능한 도구들:
            {tools_prompt}

            도구 선택 기준:
            1. 사용자의 요청 의도를 정확히 파악
            2. 각 도구의 특성과 한계를 고려
            3. 작업의 효율성과 정확성을 최적화
            4. 도구 간의 의존성이나 순서를 고려

            응답 요구사항:
            - 다음 옵션 중 정확히 하나만 선택: {tools_types}
            - 추가 설명이나 주석 없이 도구 이름만 응답
            - 선택한 도구가 현재 상황에 가장 적합한지 재확인

            주의사항:
            - 도구의 이름은 정확히 일치해야 함
            - 대소문자를 구분하여 응답"""),
            ("human", "여기에 대화 기록과 최신 메시지가 있습니다. 적절한 도구를 선택하세요:\n{input}")
        ])

        response = self.llm.invoke(prompt.format_messages(input=messages[-1].content))

        action = response.content.strip()

        # 상태 업데이트
        new_state = state.copy()
        new_state["next_step"] = action
        return new_state
    
    def _create_agent_graph(self):
        """
        에이전트의 작업 흐름을 정의하는 상태 그래프를 생성하는 함수입니다.

        Returns
        -------
        CompiledGraph
            컴파일된 상태 그래프. 다음과 같은 노드와 엣지를 포함:

            노드:
            - route: 초기 라우팅 노드
            - basic_qa: 일반 질문 답변 처리
            - data_analysis: 입력 데이터 분석 처리
            - other_tool_use: 외부 도구 선택
            - TOOLS에 정의된 각 도구별 노드

            엣지:
            1. route 노드로부터:
               - basic_qa로 연결
               - data_analysis로 연결
               - other_tool_use로 연결
               - end로 연결 (종료)

            2. other_tool_use 노드로부터:
               - 각 도구 노드로 연결

            3. 종료 엣지:
               - basic_qa → end
               - data_analysis → end
               - 각 도구 노드 → end
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("route", self.route)
        workflow.add_node("basic_qa", self.handle_basic_qa)
        workflow.add_node("data_analysis", self.handle_data_analysis)
        workflow.add_node("other_tool_use", self.tool_use)

        for tool in self.TOOLs:
            workflow.add_node(tool["name"], tool["function_name"])

        # Add edges
        workflow.set_entry_point("route")

        workflow.add_conditional_edges(
            "route",
            lambda x: x["next_step"],
            {
                "basic_qa": "basic_qa",
                "data_analysis": "data_analysis",
                "other_tool_use": "other_tool_use",
                "end": END
            },
        )

        workflow.add_conditional_edges(
            "other_tool_use", 
            lambda x: x["next_step"],
            dict((tool["name"], tool["name"]) for tool in self.TOOLs),
        )

        # 응답 생성 엣지
        workflow.add_edge("basic_qa", END)
        workflow.add_edge("data_analysis", END)

        for tool in self.TOOLs:
            workflow.add_edge(tool["name"], END)

        return workflow.compile()
    
    def alphafold_protein_structure_prediction(self, state: AgentState) -> AgentState:
        """
        AlphaFold2를 사용하여 단백질 구조를 예측하는 함수입니다.
        AWS Batch를 활용하여 Jackhmmer와 AlphaFold2 작업을 순차적으로 실행합니다.

        Parameters
        ----------
        state : AgentState
            현재 에이전트의 상태를 담고 있는 딕셔너리.
            필수 키:
            - 'proteins': BioPython SeqRecord 객체의 리스트
              각 SeqRecord는 다음을 포함:
              - id: 단백질 식별자
              - seq: 아미노산 서열
              - description: 단백질 설명 (선택사항)

        Returns
        -------
        AgentState
            업데이트된 상태 딕셔너리. 다음 키를 포함:
            - 'current_response': 작업 제출 상태 메시지
              성공 시: "All jobs are submitted successfully!"
              실패 시: "Some jobs are failed!"
        """
        try:
            for fasta in state["proteins"]:
                target_id, sequence = fasta.id, str(fasta.seq)
                try:
                    description = fasta.description
                except:
                    description = "None"
                target = BatchFoldTarget(target_id=target_id, s3_bucket=S3_BUCKET, boto_session=boto_session)
                target.add_sequence(
                    seq_id=target_id,
                    seq=sequence,
                    description=description,
                )
                target.target_id = target.target_id.split("|")[0]
                job_name = target.target_id + "_JackhmmerJob_" + datetime.now().strftime("%Y%m%d%s")

                jackhmmer_job = JackhmmerJob(
                    job_name=job_name,
                    target_id=target.target_id,
                    fasta_s3_uri=target.get_fasta_s3_uri(),
                    output_s3_uri=target.get_msas_s3_uri(),
                    boto_session=boto_session,
                    cpu=16,
                    memory=31,
                )
                jackhmmer_submission = batch_environment.submit_job(
                    jackhmmer_job, job_queue_name="CPUOnDemandJobQueue"
                )

                job_name = target.target_id + "_AlphaFold2Job_" + datetime.now().strftime("%Y%m%d%s")
                alphafold2_job = AlphaFold2Job(
                    job_name=job_name,
                    target_id=target.target_id,
                    fasta_s3_uri=target.get_fasta_s3_uri(),
                    msa_s3_uri=target.get_msas_s3_uri(),
                    output_s3_uri=target.get_predictions_s3_uri() + "/" + job_name,
                    boto_session=boto_session,
                    cpu=4,
                    memory=15,  # Why not 16? ECS needs about 1 GB for container services
                    gpu=1,
                )
                alphafold2_submission = batch_environment.submit_job(
                    alphafold2_job, job_queue_name="G4dnJobQueue", depends_on=[jackhmmer_submission]
                )

            new_state = state.copy()
            new_state["current_response"] = "All jobs are submitted successfully!"

        except:
            new_state["current_response"] = "Some jobs are failed!"

        return new_state
    
    def _convert_floats_to_decimals(self, obj):
        """
        DynamoDB compatibility를 위해 float values 를 Decimal values 로 변환 
        """
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimals(i) for i in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))
        return obj
    
    
    def add_csv_file(self, file_name: str):
        """
        LLM input에 CSV 파일 추가

        Args:
            file_name (str): Path to the CSV file
        """
        df = pd.read_csv(f"molecule/{file_name}")
        self.state["files"][file_name] = df.to_string(index=False)
        
    def add_and_save_csv_file(self, file_name: str, experiment_name: str=None, user_name: str="hasunyu"):
        """
        CSV 데이터를 DynamoDB에 추가

        Args:
            csv_file_path (str): Path to the CSV file
            experiment_name (str, optional): Custom experiment ID. If not provided, generates one
        """
        df = pd.read_csv(f"molecule/{file_name}")
        self.state["files"][file_name] = df.to_string(index=False)

        data = df.to_dict('records')
        
        data = self._convert_floats_to_decimals(data)
        
        current_date = datetime.now().strftime("%Y%m%d%s")
        
        if not experiment_name:
            experiment_id = f"Experiment_{user_name}_{current_date}"
        else:
            experiment_id = f"{experiment_name}_{user_name}_{current_date}"
            
        experiment_item = {
            'ExperimentID': experiment_id,
            'Data': data,
        }
        
        try:
            response = table.put_item(Item=experiment_item)
            print(f"Successfully uploaded experiment {experiment_id} to DynamoDB")

        except Exception as e:
            print(f"Error uploading to DynamoDB: {str(e)}")
            raise        
    
    def load_csv_file(self, experiment_id: str):
        """
        Retrieve experiment data from DynamoDB using experiment_id

        Args:
            experiment_id (str): The ID of the experiment to retrieve

        Returns:
            dict: The experiment data including ExperimentID, Data, and Type
            None: If the experiment is not found
        """
        try:
            # Get item from DynamoDB
            response = table.get_item(
                Key={
                    'ExperimentID': experiment_id
                }
            )

            # Check if item exists
            if 'Item' in response:
                experiment = response['Item']
                print(f"Successfully retrieved experiment {experiment_id}")

                # Convert the data to pandas DataFrame for easier handling
                if 'Data' in experiment:
                    data = [{k: float(v) if isinstance(v, Decimal) else v 
                            for k, v in item.items()} 
                           for item in experiment['Data']]
                    df = pd.DataFrame(data)
                
                    self.state["files"]['training.csv'] = df.to_string(index=False)
            else:
                print(f"No experiment found with ID: {experiment_id}")

        except Exception as e:
            print(f"Error retrieving experiment: {str(e)}")
            raise
    
    def chat(self, message: str):
        """
        사용자의 메시지를 처리하고 대화 상태를 업데이트하는 함수입니다.

        Parameters
        ----------
        message : str
            사용자로부터 받은 입력 메시지.
            지원되는 메시지 유형:
            - 일반적인 질문
            - 데이터 분석 요청
            - 도구 사용 요청
            - "종료" 메시지

        Returns
        -------
        dict
            업데이트된 대화 상태를 포함하는 딕셔너리.
            포함되는 키:
            - 'messages': 전체 대화 히스토리
            - 'current_response': 최신 응답 내용
            - 'next_step': 다음 실행할 작업
            다른 키들은 컨텍스트에 따라 추가될 수 있음
            (예: 'files', 'proteins' 등)
        """
        self.state["messages"].append(HumanMessage(content=message))
        self.state = self.graph.invoke(self.state)
        return self.state
    
    @staticmethod
    def list_experiments():
        """
        DynamoDB에 저장된 모든 실험 결과 조회

        Returns:
            list: List of experiment IDs
        """
        try:
            # Scan the table
            response = table.scan(
                ProjectionExpression='ExperimentID',
            )

            experiments = response['Items']

            # Handle pagination if there are more items
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression='ExperimentID',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                experiments.extend(response['Items'])

            # Sort experiments by ID
            experiments.sort(key=lambda x: x['ExperimentID'])

            # Print experiments
            print("\nStored Experiments:")
            print("-" * 50)
            print(f"{'ExperimentID':<30} {'Type':<20}")
            print("-" * 50)
            for exp in experiments:
                print(f"{exp['ExperimentID']:<30}")

            return experiments

        except Exception as e:
            print(f"Error listing experiments: {str(e)}")
            raise
        
    def add_protein(self, file_name: str):
        """
        파일로부터 FASTA 파일 load
        
        Args:
            file_name (str): FASTA file 경로
        """
        fasta_sequences = SeqIO.parse(open(f"protein/{file_name}"),"fasta")
        self.state["proteins"] = fasta_sequences
    
    def load_and_prepare_data(previous_file, current_file):
        """두 CSV 파일을 읽고 데이터를 준비"""
        df_prev = pd.read_csv(previous_file)
        df_curr = pd.read_csv(current_file)

        # SMILES를 기준으로 데이터 병합
        merged_df = pd.merge(df_prev, df_curr, 
                            on='input', 
                            suffixes=('_previous', '_current'))

        # 용해도 값을 float로 변환
        merged_df['output_previous'] = merged_df['output_previous'].astype(float)
        merged_df['output_current'] = merged_df['output_current'].astype(float)

        return merged_df

    def create_correlation_plot(merged_df, output_file='correlation_plot.png'):
        """상관관계 플롯 생성"""
        # 피어슨 상관계수 계산
        correlation = stats.pearsonr(merged_df['output_previous'], 
                                   merged_df['output_current'])
        r_squared = correlation[0]**2

        # 플롯 생성
        plt.figure(figsize=(10, 8))

        # 산점도
        sns.scatterplot(data=merged_df, 
                       x='output_previous', 
                       y='output_current', 
                       alpha=0.6)

        # 대각선 추가 (y=x 라인)
        min_val = min(merged_df['output_previous'].min(), 
                     merged_df['output_current'].min())
        max_val = max(merged_df['output_previous'].max(), 
                     merged_df['output_current'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                 'r--', label='y=x')

        # 회귀선 추가
        sns.regplot(data=merged_df, 
                    x='output_previous', 
                    y='output_current', 
                    scatter=False, 
                    color='blue', 
                    line_kws={'label': f'Regression line\nR² = {r_squared:.3f}'})

        # 그래프 스타일링
        plt.title('Correlation between Previous and Current Solubility Predictions', 
                 fontsize=14, pad=20)
        plt.xlabel('Previous Predictions', fontsize=12)
        plt.ylabel('Current Predictions', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(merged_df['output_previous'], 
                                        merged_df['output_current']))

        # 통계 정보 추가
        stats_text = (f'Statistics:\n'
                     f'R² = {r_squared:.3f}\n'
                     f'RMSE = {rmse:.3f}\n'
                     f'Pearson r = {correlation[0]:.3f}\n'
                     f'p-value = {correlation[1]:.3e}')

        plt.text(0.05, 0.95, stats_text, 
                 transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top', 
                 fontsize=10)

        # 그래프 저장
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"상관관계 플롯이 {output_file}에 저장되었습니다.")

        return r_squared, rmse, correlation