import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import base64
from io import StringIO

from models.agent import DrugDiscoveryAssistantAgent
from utils.visualization import create_correlation_plot
from config.config import AWS_CONFIG

def initialize_session_state():
    """세션 상태 초기화"""
    if 'agent' not in st.session_state:
        st.session_state.agent = DrugDiscoveryAssistantAgent(
            aws_region=AWS_CONFIG["region"],
            model_name=AWS_CONFIG["model_name"]
        )
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'experiment_list' not in st.session_state:
        st.session_state.experiment_list = None
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None

def handle_file_upload(uploaded_file, file_type, save_to_dynamodb=True):
    """파일 업로드 처리"""
    if uploaded_file is None:
        return

    try:
        if file_type in ['train', 'test']:
            # CSV 파일 처리
            filename = f"{file_type}.csv"
            save_path = Path('molecule') / filename
            save_path.parent.mkdir(exist_ok=True)
            
            # StringIO 객체인 경우와 UploadedFile 객체인 경우를 구분하여 처리
            if isinstance(uploaded_file, StringIO):
                with open(save_path, 'w') as f:
                    f.write(uploaded_file.getvalue())
            else:
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
            if file_type == 'train':
                if save_to_dynamodb:
                    st.session_state.agent.add_and_save_csv_file(
                        filename,
                        "Solubility_Training"
                    )
                    st.success(f"{filename} 파일이 성공적으로 업로드되고 DynamoDB에 저장되었습니다.")
                else:
                    st.session_state.agent.add_csv_file(filename)
                    st.success(f"{filename} 파일이 성공적으로 로드되었습니다.")
            else:
                st.session_state.agent.add_csv_file(filename)
                st.success(f"{filename} 파일이 성공적으로 업로드되었습니다.")
                
        elif file_type == 'fasta':
            # FASTA 파일 처리
            save_path = Path('protein') / uploaded_file.name
            save_path.parent.mkdir(exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
                
            st.session_state.agent.add_protein(uploaded_file.name)
            st.success(f"FASTA 파일이 성공적으로 업로드되었습니다.")
            
    except Exception as e:
        st.error(f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

def handle_experiment_data(selected_experiment):
    """선택된 실험 데이터 처리"""
    with st.spinner(f"실험 데이터 {selected_experiment} 로딩 중..."):
        try:
            st.session_state.current_experiment_id = selected_experiment
            csv_content = st.session_state.agent.load_csv_file(selected_experiment)
            
            if csv_content is None:
                st.error("데이터를 불러올 수 없습니다.")
                return
            
            # 데이터 미리보기
            try:
                df = pd.read_csv(StringIO(csv_content))
                st.write("데이터 미리보기:")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"데이터 미리보기 중 오류 발생: {str(e)}")
            
            # molecule 디렉토리 확인 및 생성
            molecule_dir = Path('molecule')
            molecule_dir.mkdir(exist_ok=True)
            
            # train.csv 파일 저장
            train_file = molecule_dir / 'train.csv'
            with open(train_file, 'w') as f:
                f.write(csv_content)
            
            # agent에 파일 추가
            st.session_state.agent.add_csv_file('train.csv')
            st.success(f"실험 데이터 {selected_experiment}가 성공적으로 로드되었습니다.")
            
            if not train_file.exists():
                st.error(f"파일이 생성되지 않았습니다: {train_file}")
            else:
                st.success(f"파일이 성공적으로 생성되었습니다: {train_file}")
                
        except Exception as e:
            st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
            import traceback
            st.error(f"상세 에러:\n```\n{traceback.format_exc()}\n```")
        
def display_experiment_list():
    """DynamoDB에 저장된 실험 목록 표시"""
    st.markdown("### 💾 저장된 실험 데이터")
    
    try:
        # 실험 목록 가져오기
        experiments = st.session_state.agent.list_experiments()
        
        if not experiments:
            st.info("저장된 실험이 없습니다.")
            return
            
        # 실험 목록을 DataFrame으로 변환하여 보기 좋게 표시
        exp_df = pd.DataFrame(experiments)
        if 'Timestamp' in exp_df.columns:
            exp_df['Timestamp'] = pd.to_datetime(exp_df['Timestamp'])
            exp_df = exp_df.sort_values('Timestamp', ascending=False)
        
        # 실험 선택을 위한 radio 버튼 생성
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_experiment = st.radio(
                "실험을 선택하세요:",
                options=exp_df['ExperimentID'].tolist(),
                index=None,
                key="experiment_selector"
            )
        
        with col2:
            if st.button("🔄 새로고침", key="refresh_button"):
                st.session_state.current_experiment_id = None
                st.rerun()
        
        # 선택된 실험이 있고, 이전 선택과 다른 경우에만 처리
        if selected_experiment and (
            'current_experiment_id' not in st.session_state or 
            selected_experiment != st.session_state.current_experiment_id
        ):
            with st.spinner(f"실험 데이터 {selected_experiment} 로딩 중..."):
                try:
                    # 현재 선택된 실험 ID 저장
                    st.session_state.current_experiment_id = selected_experiment
                    
                    # DynamoDB에서 데이터 로드
                    csv_content = st.session_state.agent.load_csv_file(selected_experiment)
                    
                    if csv_content is None:
                        st.error(selected_experiment)
                        st.error("데이터를 불러올 수 없습니다.")
                        return
                    
                    # 데이터 미리보기
                    try:
                        df = pd.read_csv(StringIO(csv_content))
                        st.write("데이터 미리보기:")
                        st.dataframe(df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"데이터 미리보기 중 오류 발생: {str(e)}")
                    
                    # molecule 디렉토리 확인 및 생성
                    molecule_dir = Path('molecule')
                    molecule_dir.mkdir(exist_ok=True)
                    
                    # train.csv 파일 저장
                    train_file = molecule_dir / 'train.csv'
                    with open(train_file, 'w') as f:
                        f.write(csv_content)
                    
                    # agent에 파일 추가
                    st.session_state.agent.add_csv_file('train.csv')
                    
                    st.success(f"실험 데이터 {selected_experiment}가 성공적으로 로드되었습니다.")
                    
                    # 파일 존재 확인
                    if not train_file.exists():
                        st.error(f"파일이 생성되지 않았습니다: {train_file}")
                    else:
                        st.success(f"파일이 성공적으로 생성되었습니다: {train_file}")
                        
                except Exception as e:
                    st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
                    import traceback
                    st.error(f"상세 에러:\n```\n{traceback.format_exc()}\n```")
    
    except Exception as e:
        st.error(f"실험 목록 로드 중 오류가 발생했습니다: {str(e)}")
        import traceback
        st.error(f"상세 에러:\n```\n{traceback.format_exc()}\n```")

def display_visualization():
    """예측 결과 시각화"""
    try:
        if os.path.exists("molecule/test_prediction.csv") and os.path.exists("molecule/test_gt.csv"):
            df_prev = pd.read_csv("molecule/test_gt.csv")
            df_curr = pd.read_csv("molecule/test_prediction.csv")
            
            fig, r_squared, rmse, correlation = create_correlation_plot(df_prev, df_curr)
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{r_squared:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.3f}")
            with col3:
                st.metric("Correlation", f"{correlation:.3f}")
            
            # 예측 결과 다운로드 버튼
            csv = df_curr.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">예측 결과 다운로드</a>'
            st.markdown(href, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"시각화 중 오류가 발생했습니다: {str(e)}")

def display_chat_interface():
    """채팅 인터페이스 표시"""
    st.title("Drug Discovery Assistant")
    
    with st.sidebar:
        # 1. 기능 소개 섹션
        st.markdown("## 🤖 기능 가이드")
        with st.expander("사용 가능한 기능 보기", expanded=True):
            st.markdown("""
            1. **🔍 기본 정보 문의**
               - 단백질과 관련된 기본적인 질문에 답변
               - 예시: _"EGFR 단백질의 기본 정보를 알려줘"_
            
            2. **📊 분자 특성 예측**
               - Training 데이터와 Test 데이터를 업로드하여 특성 예측
               - DynamoDB에 저장된 데이터 활용 가능
               - 예시: _"첨부한 train.csv 파일의 정보를 바탕으로 test.csv 파일의 분자들의 특성을 예측해줘"_
            
            3. **🧬 단백질 구조 예측**
               - FASTA 파일을 업로드하여 AlphaFold2로 구조 예측
               - 예시: _"첨부한 fasta 파일에 존재하는 단백질 서열의 구조를 예측해줘"_
            """)
        
        st.markdown("---")
        
        # 2. 학습 데이터 관리 섹션
        st.markdown("## 📚 학습 데이터 관리")
        
        # 실험 데이터 관리 섹션
        with st.expander("💾 저장된 실험 데이터", expanded=True):
            try:
                # 새로고침 버튼
                if st.button("🔄 새로고침"):
                    st.session_state.current_experiment_id = None
                    st.rerun()
                
                # 실험 목록 가져오기
                experiments = st.session_state.agent.list_experiments()
                
                if not experiments:
                    st.info("저장된 실험이 없습니다.")
                else:
                    # 실험 목록을 DataFrame으로 변환
                    exp_df = pd.DataFrame(experiments)
                    if 'Timestamp' in exp_df.columns:
                        exp_df['Timestamp'] = pd.to_datetime(exp_df['Timestamp'])
                        exp_df = exp_df.sort_values('Timestamp', ascending=False)
                    
                    # 실험 선택
                    selected_experiment = st.radio(
                        "실험을 선택하세요:",
                        options=exp_df['ExperimentID'].tolist(),
                        index=None
                    )
                    
                    # 선택된 실험 처리
                    if selected_experiment and (
                        'current_experiment_id' not in st.session_state or 
                        selected_experiment != st.session_state.current_experiment_id
                    ):
                        handle_experiment_data(selected_experiment)
                        
            except Exception as e:
                st.error(f"실험 목록 로드 중 오류가 발생했습니다: {str(e)}")
                import traceback
                st.error(f"상세 에러:\n```\n{traceback.format_exc()}\n```")
        
        # 파일 업로더 섹션
        uploaded_train = st.file_uploader(
            "신규 학습 데이터 업로드 (CSV)",
            type=['csv'],
            key='train',
            help="분자 구조와 특성이 포함된 학습 데이터"
        )
        
        st.markdown("---")
        
        # 3. 테스트 데이터 섹션
        st.markdown("## 🧪 테스트 데이터")
        uploaded_test = st.file_uploader(
            "테스트 데이터 (CSV)",
            type=['csv'],
            key='test',
            help="특성을 예측할 분자 구조 데이터"
        )
        
        st.markdown("---")
        
        # 4. 단백질 데이터 섹션
        st.markdown("## 🧬 단백질 데이터")
        uploaded_fasta = st.file_uploader(
            "단백질 서열 데이터 (FASTA)",
            type=['fasta', 'fa'],
            key='fasta',
            help="구조를 예측할 단백질 서열 데이터"
        )
        
        # 파일 업로드 처리
        for upload, file_type in [
            (uploaded_train, 'train'),
            (uploaded_test, 'test'),
            (uploaded_fasta, 'fasta')
        ]:
            if upload:
                with st.spinner(f'{file_type.title()} 파일 처리 중...'):
                    handle_file_upload(upload, file_type)

        # 업로드된 파일 상태 표시
        if any([uploaded_train, uploaded_test, uploaded_fasta]):
            with st.expander("📁 업로드된 파일 상태", expanded=False):
                for file, name in [
                    (uploaded_train, "학습 데이터"),
                    (uploaded_test, "테스트 데이터"),
                    (uploaded_fasta, "단백질 데이터")
                ]:
                    if file:
                        st.success(f"✅ {name} 업로드 완료")

    # 메인 채팅 영역
    chat_container = st.container()
    with chat_container:
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("메시지를 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("생각하는 중..."):
                    response = st.session_state.agent.chat(prompt)
                    
                    if response.get('current_response'):
                        response_text = response['current_response']
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                        # 데이터 분석 결과가 있고, 특정 조건을 만족할 때만 시각화 표시
                        if os.path.exists("molecule/test_prediction.csv") and \
                           os.path.exists("molecule/test_gt.csv") and \
                           any(keyword in prompt.lower() for keyword in 
                               ["분자", "상관관계", "correlation", "시각화", "그래프", "plot", "플롯", "결과", "예측"]) and \
                           not any(keyword in prompt.lower() for keyword in 
                                 ["fasta", "단백질", "alphafold", "알파폴드"]):
                            with st.expander("예측 결과 시각화", expanded=True):
                                display_visualization()
            
        

    # 메인 채팅 영역
    chat_container = st.container()
    with chat_container:
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("메시지를 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("생각하는 중..."):
                    response = st.session_state.agent.chat(prompt)
                    
                    if response.get('current_response'):
                        response_text = response['current_response']
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                        # 데이터 분석 결과가 있고, 특정 조건을 만족할 때만 시각화 표시
                        if os.path.exists("molecule/test_prediction.csv") and \
                           os.path.exists("molecule/test_gt.csv") and \
                           any(keyword in prompt.lower() for keyword in 
                               ["분자", "상관관계", "correlation", "시각화", "그래프", "plot", "플롯", "결과", "예측"]) and \
                           not any(keyword in prompt.lower() for keyword in 
                                 ["fasta", "단백질", "alphafold", "알파폴드"]):
                            with st.expander("예측 결과 시각화", expanded=True):
                                display_visualization()
                                
def main():
    st.set_page_config(
        page_title="Drug Discovery Assistant",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    display_chat_interface()

if __name__ == "__main__":
    main()