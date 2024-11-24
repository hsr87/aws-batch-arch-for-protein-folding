import os
import sys
import json
from decimal import Decimal
import pandas as pd
from typing import Any, Dict, List, Sequence, TypedDict
from datetime import datetime

import boto3
from botocore.config import Config
from Bio import SeqIO

from batchfold.batchfold_environment import BatchFoldEnvironment
from batchfold.batchfold_target import BatchFoldTarget
from batchfold.jackhmmer_job import JackhmmerJob
from batchfold.alphafold2_job import AlphaFold2Job
from batchfold.esmfold_job import ESMFoldJob
from batchfold.rfdiffusion_job import RFDiffusionJob

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import graph, END, StateGraph
from langchain.prompts import ChatPromptTemplate

class AgentState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    next_step: str | None
    files: Dict[str, Any]
    current_response: str | None
    proteins: BatchFoldTarget | None

class DrugDiscoveryAssistantAgent:
    def __init__(self, aws_region="us-west-2", model_name="anthropic.claude-3-5-sonnet-20241022-v2:0"):
        self.aws_region = aws_region
        self.model_name = model_name
        self._initialize_aws_clients()
        self.state = {
            "messages": [],
            "next_step": None,
            "files": {},
            "current_response": None,
            "proteins": None,
        }
        self.TOOLs = [
            {
                "name": "alphafold_protein_structure_prediction",
                "function_name": self.alphafold_protein_structure_prediction,
                "description": """
                                AlphaFold를 사용하여 단백질 구조를 예측합니다.
                                이 도구는 단백질 시퀀스를 입력받아 3D 구조를 예측합니다.

                                입력:
                                - protein_sequence (str): 아미노산 서열 (FASTA 형식의 문자열)

                                출력:
                                - predicted_structure: PDB 형식의 3D 구조 예측 결과

                                사용 조건:
                                - 단백질 구조 예측 시 기본적으로 첫 번째로 시도해야 하는 도구입니다.
                                - MSA(Multiple Sequence Alignment)가 사용 가능한 경우 더 정확한 결과를 제공합니다.
                """,
                "input_file": "fasta",
            },
            {
                "name": "esmfold_protein_structure_prediction",
                "function_name": self.esmfold_protein_structure_prediction,
                "description": """
                                ESMFold를 사용하여 단백질 구조를 예측합니다.
                                MSA(Multiple Sequence Alignment) 없이도 단일 시퀀스만으로 구조 예측이 가능합니다.

                                입력:
                                - protein_sequence (str): 아미노산 서열 (FASTA 형식의 문자열)

                                출력:
                                - predicted_structure: PDB 형식의 3D 구조 예측 결과

                                사용 조건:
                                - MSA(Multiple Sequence Alignment)를 사용하지 않고 구조를 예측하는 경우 사용합니다.
                                - 비교적 빠른 예측이 필요한 경우에도 사용할 수 있습니다.
                """,
                "input_file": "fasta",
            },
            {
                "name": "rfdiffusion_protein_generation",
                "function_name": self.rfdiffusion_protein_generation,
                "description": """
                                RFDiffusion을 사용하여 입력받은 단백질의 Binder 3D 구조를 생성해줘.

                                입력:
                                - protein_pdb (str): PDB 형식의 3D 구조

                                출력:
                                - predicted_binder_structure: PDB 형식의 3D 구조 예측 결과

                                사용 조건:
                                - 단백질 binder 생성 시 사용합니다.

                """,
                "input_file": "pdb",
            },
        ]
        self.llm = self._make_llm()
        self.graph = self._create_agent_graph()

    def _initialize_aws_clients(self):
        """AWS 클라이언트 초기화"""
        bedrock_config = Config(
            connect_timeout=300, 
            read_timeout=300, 
            retries={"max_attempts": 10}
        )
        self.bedrock_rt = boto3.client(
            "bedrock-runtime", 
            region_name=self.aws_region, 
            config=bedrock_config
        )
        
        self.boto_session = boto3.session.Session()
        self.batch_environment = BatchFoldEnvironment(boto_session=self.boto_session)
        self.S3_BUCKET = self.batch_environment.default_bucket
        
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('Experiment')

    def _make_llm(self, temperature=0, max_tokens=None):
        """LLM 인스턴스 생성"""
        return ChatBedrockConverse(
            client=self.bedrock_rt,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _convert_floats_to_decimals(self, obj):
        """DynamoDB를 위한 float to Decimal 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimals(i) for i in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))
        return obj

    def add_csv_file(self, file_name: str):
        """CSV 파일 추가"""
        df = pd.read_csv(f"molecule/{file_name}")
        self.state["files"][file_name] = df.to_string(index=False)

    def add_and_save_csv_file(self, file_name: str, experiment_name: str = None, user_name: str = "hasunyu"):
        """CSV 파일 추가 및 DynamoDB에 저장"""
        df = pd.read_csv(f"molecule/{file_name}")
        self.state["files"][file_name] = df.to_string(index=False)

        data = df.to_dict('records')
        
        data = self._convert_floats_to_decimals(data)
        
        current_date = datetime.now().strftime("%Y%m%d%s")
        
        experiment_id = f"{experiment_name}_{user_name}_{current_date}" if experiment_name else f"Experiment_{user_name}_{current_date}"
            
        experiment_item = {
            'ExperimentID': experiment_id,
            'Data': data,
        }
        
        try:
            self.table.put_item(Item=experiment_item)
            print(f"Successfully uploaded experiment {experiment_id} to DynamoDB")
        except Exception as e:
            print(f"Error uploading to DynamoDB: {str(e)}")
            raise

    def load_csv_file(self, experiment_id: str):
        """
        DynamoDB에서 실험 데이터를 로드하여 DataFrame으로 반환

        Parameters
        ----------
        experiment_id : str
            로드할 실험의 ID

        Returns
        -------
        pandas.DataFrame or None
            로드된 데이터를 담은 DataFrame, 실패시 None
        """
        try:
            response = self.table.get_item(
                Key={'ExperimentID': experiment_id}
            )

            if 'Item' in response:
                experiment = response['Item']
                print(f"Successfully retrieved experiment {experiment_id}")

                if 'Data' in experiment:
                    data = [{k: float(v) if isinstance(v, Decimal) else v 
                            for k, v in item.items()} 
                           for item in experiment['Data']]
                    return pd.DataFrame(data)

            print(f"No experiment found with ID: {experiment_id}")
            return None

        except Exception as e:
            print(f"Error retrieving experiment: {str(e)}")

    @staticmethod
    def list_experiments():
        """DynamoDB에 저장된 모든 실험 목록 조회"""
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table('Experiment')
            
            response = table.scan(
                ProjectionExpression='ExperimentID',
            )
            experiments = response['Items']

            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression='ExperimentID',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                experiments.extend(response['Items'])

            experiments.sort(key=lambda x: x['ExperimentID'])

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
        """FASTA 파일에서 단백질 시퀀스 로드"""
        fasta_sequences = SeqIO.parse(open(f"protein/{file_name}"), "fasta")
        self.state["proteins"] = fasta_sequences
    
    def add_3d_protein(self, file_name: str):
        """PDB 파일에서 단백질 구조 load"""
        fasta_sequences = SeqIO.parse(open(f"protein/{file_name}"), "fasta")
        self.state["proteins"] = fasta_sequences

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
            - tool_use: 외부 도구 선택
            - TOOLS에 정의된 각 도구별 노드

            엣지:
            1. route 노드로부터:
               - basic_qa로 연결
               - data_analysis로 연결
               - tool_use로 연결
               - end로 연결 (종료)

            2. tool_use 노드로부터:
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
        workflow.add_node("tool_use", self.tool_use)

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
                "tool_use": "tool_use",
                "end": END
            },
        )

        workflow.add_conditional_edges(
            "tool_use", 
            lambda x: x["next_step"],
            dict((tool["name"], tool["name"]) for tool in self.TOOLs),
        )

        # 응답 생성 엣지
        workflow.add_edge("basic_qa", END)
        workflow.add_edge("data_analysis", END)

        for tool in self.TOOLs:
            workflow.add_edge(tool["name"], END)

        return workflow.compile()
    
    def route(self, state: AgentState) -> Dict[str, str]:
        """
        사용자의 입력을 분석하여 적절한 다음 단계의 작업을 결정하는 라우팅 함수입니다.
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

            3. "tool_use" - 다음과 같은 경우 선택:
               - AlphaFold 등 외부 도구 사용 필요
               - 단백질 구조 예측 요청

            반드시 다음 세 단어 중 하나만 답변으로 제시하세요: basic_qa, data_analysis, tool_use
            추가 설명이나 다른 텍스트를 포함하지 마세요.
            """),
            ("human", "다음 사용자 메시지를 분석하여 적절한 작업을 선택하세요:\n{input}")
        ])

        response = self.llm.invoke(prompt.format_messages(input=messages[-1].content))
        action = response.content.strip().lower()

        # 응답에서 실제 작업만 추출
        if "basic_qa" in action:
            action = "basic_qa"
        elif "data_analysis" in action:
            action = "data_analysis"
        elif "tool_use" in action:
            action = "tool_use"

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
              - 'train.csv': 학습 데이터
              - 'test.csv': 테스트 데이터

        Returns
        -------
        AgentState
            업데이트된 상태 딕셔너리. 다음 키를 포함:
            - 'current_response': LLM이 생성한 예측 결과

        Raises
        ------
        FileNotFoundError
            필요한 파일(train.csv 또는 test.csv)이 없는 경우

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
        ...         "train.csv": "SMILES,solubility\\nC1=CC=CC=C1,2.3\\n",
        ...         "test.csv": "SMILES\\nC1=CC=CN=C1\\n"
        ...     }
        ... }
        >>> result = handle_data_analysis(state)
        >>> print("예측 결과가 저장되었습니다.")
        """
        messages = state["messages"]
        files = state["files"]

        required_files = {"train.csv", "test.csv"}
        missing_files = required_files - files.keys()

        if missing_files:
            raise FileNotFoundError(f"필요한 파일이 없습니다: {', '.join(missing_files)}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 신약 개발 분야의 데이터 분석 전문가입니다. 
            분자 데이터를 분석하여 특성을 예측하는 작업을 수행해야 합니다.

            필수 요구사항:
            1. 출력 형식
               - 반드시 다음 형식의 CSV 데이터만 출력:
               input,output,property name
               [SMILES],[예측값],[특성이름]

            2. 컬럼 설명
               - input: SMILES 문자열 (테스트 데이터와 동일한 형식)
               - output: 예측된 특성값 (실수)
               - property name: 특성의 이름 (예: Solubility)

            3. 주의사항
               - CSV 헤더는 정확히 'input,output,property name'이어야 함
               - 추가 설명이나 주석 없이 CSV 데이터만 출력
               - 모든 테스트 분자에 대해 예측값 제공
               - 각 행은 하나의 분자에 대한 예측만 포함
            """),
            ("human", """Training data: {training_data}\n
            Test data: {test_data}\n
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            input = messages[-1].content,
            training_data = files['train.csv'],
            test_data = files['test.csv'],
            )
        )
    
        def _parse_claude_response(response: str) -> List[Dict]:
            """
            LLM의 응답에서 분자와 예측된 특성을 추출

            Parameters
            ----------
            response : str
                LLM이 생성한 CSV 형식의 응답

            Returns
            -------
            List[Dict]
                파싱된 예측 결과의 리스트

            Notes
            -----
            응답 형식이 올바르지 않은 경우를 처리하기 위한 여러 방법을 시도
            """
            try:
                # 응답에서 CSV 부분만 추출
                import re
                # CSV 형식의 데이터를 찾기 위한 패턴
                csv_pattern = r'(?:^|\n)(?:input|SMILES|molecule).*?(?:\n.*?)*$'
                match = re.search(csv_pattern, response, re.MULTILINE)

                if match:
                    csv_content = match.group(0)
                else:
                    csv_content = response

                # CSV 파싱 시도
                from io import StringIO
                df = pd.read_csv(StringIO(csv_content))

                # 컬럼명 표준화
                column_mapping = {
                    'SMILES': 'input',
                    'molecule': 'input',
                    'prediction': 'output',
                    'value': 'output',
                    'property': 'property name',
                    'type': 'property name'
                }

                df = df.rename(columns=column_mapping)

                # 필수 컬럼 확인
                required_columns = {'input', 'output', 'property name'}
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Required columns missing. Found: {df.columns}")

                return df.to_dict('records')

            except Exception as e:
                print(f"CSV 파싱 오류: {str(e)}")
                print(f"원본 응답:\n{response}")

                # 대체 파싱 시도
                try:
                    # 테스트 파일에서 입력 분자 가져오기
                    test_df = pd.read_csv("molecule/test.csv")
                    inputs = test_df['input'].tolist() if 'input' in test_df.columns else test_df.iloc[:, 0].tolist()

                    # 숫자만 추출
                    import re
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                    numbers = [float(n) for n in numbers]

                    # 결과 생성
                    results = []
                    for i, smiles in enumerate(inputs):
                        if i < len(numbers):
                            results.append({
                                'input': smiles,
                                'output': numbers[i],
                                'property name': 'Solubility'
                            })

                    if not results:
                        raise ValueError("No valid predictions could be extracted")

                    return results

                except Exception as backup_e:
                    print(f"대체 파싱도 실패: {str(backup_e)}")
                    raise ValueError("예측 결과를 파싱할 수 없습니다") from backup_e

        def _save_predictions_to_csv(predictions: List[Dict], filename: str = "molecule/test_prediction.csv"):
            """예측 결과를 CSV 파일로 저장"""
            df = pd.DataFrame(predictions)
            df.to_csv(filename, index=False)
            print(f"예측 결과가 {filename}에 저장되었습니다.")

        predictions = _parse_claude_response(response.content)
        _save_predictions_to_csv(predictions)

        new_state = state.copy()
        new_state["current_response"] = response.content
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
              성공 시: "모든 단백질에 대해 구조 예측이 시작되었습니다!"
              실패 시: "단백질 구조 예측이 실패 했습니다."
        """
        try:
            for fasta in state["proteins"]:
                target_id, sequence = fasta.id, str(fasta.seq)
                try:
                    description = fasta.description
                except:
                    description = "None"
                
                target_id = target_id.split("_")[0]
                target = BatchFoldTarget(target_id=target_id, s3_bucket=self.S3_BUCKET, boto_session=self.boto_session)
                target.add_sequence(
                    seq_id=target_id,
                    seq=sequence,
                    description=description,
                )
                
                job_name = target.target_id + "_JackhmmerJob_" + datetime.now().strftime("%Y%m%d%s")

                jackhmmer_job = JackhmmerJob(
                    job_name=job_name,
                    target_id=target.target_id,
                    fasta_s3_uri=target.get_fasta_s3_uri(),
                    output_s3_uri=target.get_msas_s3_uri(),
                    boto_session=self.boto_session,
                    cpu=16,
                    memory=31,
                )
                jackhmmer_submission = self.batch_environment.submit_job(
                    jackhmmer_job, job_queue_name="CPUOnDemandJobQueue"
                )

                job_name = target.target_id + "_AlphaFold2Job_" + datetime.now().strftime("%Y%m%d%s")
                alphafold2_job = AlphaFold2Job(
                    job_name=job_name,
                    target_id=target.target_id,
                    fasta_s3_uri=target.get_fasta_s3_uri(),
                    msa_s3_uri=target.get_msas_s3_uri(),
                    output_s3_uri=target.get_predictions_s3_uri() + "/" + job_name,
                    boto_session=self.boto_session,
                    cpu=4,
                    memory=15,  # Why not 16? ECS needs about 1 GB for container services
                    gpu=1,
                )
                alphafold2_submission = self.batch_environment.submit_job(
                    alphafold2_job, job_queue_name="G4dnJobQueue", depends_on=[jackhmmer_submission]
                )

            new_state = state.copy()
            new_state["current_response"] = f"모든 단백질에 대해 구조 예측이 시작되었습니다!\nYour result will be saved in {target.get_predictions_s3_uri() + '/' + job_name}"
    
        except Exception as e:
            new_state = state.copy()
            new_state["current_response"] = f"단백질 구조 예측이 실패 했습니다.\n에러 메시지: {str(e)}"

        return new_state
    
    def esmfold_protein_structure_prediction(self, state: AgentState) -> AgentState:
        """
        ESMFold를 사용하여 MSA 정보 없이 단백질 구조를 예측하는 함수입니다.
        AWS Batch를 활용하여 ESMFold 작업을 실행합니다.

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
              성공 시: "모든 단백질에 대해 구조 예측이 시작되었습니다!"
              실패 시: "단백질 구조 예측이 실패 했습니다."
        """
        try:
            for fasta in state["proteins"]:
                target_id, sequence = fasta.id, str(fasta.seq)
                try:
                    description = fasta.description
                except:
                    description = "None"
                
                target_id = target_id.split("_")[0]
                target = BatchFoldTarget(target_id=target_id, s3_bucket=self.S3_BUCKET, boto_session=self.boto_session)
                target.add_sequence(
                    seq_id=target_id,
                    seq=sequence,
                    description=description,
                )
                
                job_name = target.target_id + "_ESMFoldJob_" + datetime.now().strftime("%Y%m%d%s")
                esmfold_job = ESMFoldJob(
                    job_name=job_name,
                    target_id=target.target_id,
                    fasta_s3_uri=target.get_fasta_s3_uri(),
                    output_s3_uri=target.get_predictions_s3_uri() + "/" + job_name,
                    boto_session=self.boto_session,
                    cpu=8,
                    memory=31,  # Why not 32? ECS needs about 1 GB for container services
                    gpu=1,
                )
                esmfold_submission = self.batch_environment.submit_job(
                    esmfold_job, job_queue_name="G4dnJobQueue"
                )

            new_state = state.copy()
            new_state["current_response"] = f"모든 단백질에 대해 구조 예측이 시작되었습니다!\n단백질 구조 예측 결과는 다음의 경로에 저장됩니다:\n{target.get_predictions_s3_uri() + '/' + job_name}"
    
        except Exception as e:
            new_state = state.copy()
            new_state["current_response"] = f"단백질 구조 예측이 실패 했습니다.\n에러 메시지: {str(e)}"

        return new_state
    
    def rfdiffusion_protein_generation(self, state: AgentState) -> AgentState:
        """
        RFDiffusion을 사용하여 단백질 구조를 생성합니다.
        AWS Batch를 활용하여 RFDiffusion 작업을 실행합니다. 현재는 "Insulin" Example 만 제공합니다. 향후 contigmap.contigs, ppi.hotspot_res 등의 수정이 필요합니다.

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
              성공 시: "Insulin binder 예측이 성공했습니다!"
              실패 시: "Insulin binder 예측이 실패 했습니다."
        """
        try:
            target_id = "BINDER-" + datetime.now().strftime("%Y%m%d%s")
            target = BatchFoldTarget(target_id=target_id, s3_bucket=self.S3_BUCKET, boto_session=self.boto_session)
            
            JOB_QUEUE = "G4dnJobQueue"

            target.upload_pdb("protein/target.pdb")
            
            NUM_DESIGNS = 4

            params = {
                "contigmap.contigs": "[A1-150/0 70-100]", 
                "ppi.hotspot_res": "[A59,A83,A91]", 
                "inference.num_designs": NUM_DESIGNS, 
                "denoiser.noise_scale_ca": 0, 
                "denoiser.noise_scale_frame": 0,
            }

            job_name = "RFDiffusionJob" + datetime.now().strftime("%Y%m%d%s")

            rfdiffusion_job = RFDiffusionJob(
                boto_session=self.boto_session,
                job_name=job_name,
                input_s3_uri=os.path.join(target.get_pdbs_s3_uri(), os.path.basename("target.pdb")),
                output_s3_uri=target.get_predictions_s3_uri() + "/" + job_name,
                params=params,
            )
            print(f"Submitting {job_name}")
            rfdiffusion_submission = self.batch_environment.submit_job(rfdiffusion_job, job_queue_name=JOB_QUEUE)
            
            new_state = state.copy()
            new_state["current_response"] = f"모든 단백질 binder 생성이 시작되었습니다!\nYour result will be saved in {target.get_predictions_s3_uri() + '/' + job_name}"
    
        except Exception as e:
            new_state = state.copy()
            new_state["current_response"] = f"단백질 binder 생성이 실패 했습니다.\n에러 메시지: {str(e)}"

        return new_state