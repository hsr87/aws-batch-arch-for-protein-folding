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
from batchfold.utils.utils import *

from batchfold.utils import protein
from batchfold.utils import residue_constants
import numpy as np

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
    if 'file_upload_state' not in st.session_state:
        st.session_state.file_upload_state = {
            'train': None,
            'test': None,
            'fasta': None
        }
    if 'experiment_refresh_state' not in st.session_state:
        st.session_state.experiment_refresh_state = 0
        
def handle_file_upload(uploaded_file, file_type, save_to_dynamodb=True):
    """파일 업로드 처리"""
    if uploaded_file is None:
        return

    try:
        if file_type in ['train', 'test']:
            filename = f"{file_type}.csv"
            save_path = Path('molecule') / filename
            save_path.parent.mkdir(exist_ok=True)
            
            # 파일 저장
            if isinstance(uploaded_file, StringIO):
                content = uploaded_file.getvalue()
                with open(save_path, 'w') as f:
                    f.write(content)
            else:
                content = uploaded_file.getvalue()
                with open(save_path, 'wb') as f:
                    f.write(content)
            
            # 파일 처리
            if file_type == 'train':
                st.session_state.agent.add_csv_file(filename)
                if save_to_dynamodb and not isinstance(uploaded_file, StringIO):
                    # 새로운 파일 업로드일 때만 DynamoDB에 저장
                    if 'last_uploaded_train' not in st.session_state or \
                       st.session_state.last_uploaded_train != uploaded_file:
                        st.session_state.agent.add_and_save_csv_file(
                            filename,
                            "Solubility_Training"
                        )
                        st.session_state.last_uploaded_train = uploaded_file
                        st.success(f"{filename} 파일이 성공적으로 업로드되고 DynamoDB에 저장되었습니다.")
                else:
                    st.success(f"{filename} 파일이 성공적으로 로드되었습니다.")
            else:
                st.session_state.agent.add_csv_file(filename)
                st.success(f"{filename} 파일이 성공적으로 업로드되었습니다.")
                
        elif file_type == 'fasta':
            save_path = Path('protein') / uploaded_file.name
            save_path.parent.mkdir(exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
                
            st.session_state.agent.add_protein(uploaded_file.name)
            st.success(f"FASTA 파일이 성공적으로 업로드되었습니다.")
            
    except Exception as e:
        st.error(f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

def display_experiment_list():
    """DynamoDB에 저장된 실험 목록 표시"""
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None
        
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("💾 **저장된 실험 목록**")
    with col2:
        if st.button("🔄", key="refresh_experiments_button"):
            st.session_state.experiment_refresh_state += 1
            st.session_state.current_experiment_id = None
            
    # 실험 목록 가져오기
    experiments = st.session_state.agent.list_experiments()
    st.session_state.experiment_list = experiments
    
    if experiments:
        experiment_ids = [exp['ExperimentID'] for exp in experiments]
        selected_experiment = st.selectbox(
            "저장된 실험 선택",
            options=experiment_ids,
            index=None,
            placeholder="실험을 선택하세요...",
            key=f"experiment_selector_{st.session_state.experiment_refresh_state}"
        )
        
        if selected_experiment and selected_experiment != st.session_state.current_experiment_id:
            st.session_state.current_experiment_id = selected_experiment
            try:
                df = st.session_state.agent.load_csv_file(selected_experiment)
                if df is not None:
                    save_path = Path('molecule/train.csv')
                    save_path.parent.mkdir(exist_ok=True)
                    df.to_csv(save_path, index=False)
                    
                    st.session_state.agent.state["files"]['train.csv'] = df.to_string(index=False)
                    st.success(f"실험 데이터 {selected_experiment}가 성공적으로 로드되었습니다.")
            except Exception as e:
                st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")

def visualization_pdb(pdb_file, show_sidechains=False, width=800, height=600):
    """PDB 구조를 시각화하고 HTML로 반환하는 함수"""
    try:
        with open(pdb_file) as f:
            best_pdb = f.read()
        target_protein = protein.from_pdb_string(best_pdb)
        plddt_list = target_protein.b_factors[:, 0]
        atom_mask = target_protein.atom_mask
        banded_b_factors = []
        for plddt in plddt_list:
            for idx, (min_val, max_val, *_) in enumerate(residue_constants.PLDDT_BANDS):
                if plddt >= min_val and plddt <= max_val:
                    banded_b_factors.append(idx)
                    break
        banded_b_factors = np.array(banded_b_factors)[:, None] * atom_mask
        to_visualize_pdb = overwrite_b_factors(best_pdb, banded_b_factors)
        
        # Color the structure by per-residue pLDDT
        color_map = {i: bands[2] for i, bands in enumerate(residue_constants.PLDDT_BANDS)}
        
        # HTML 템플릿 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        </head>
        <body>
            <div id="viewport" style="width:{width}px; height:{height}px;"></div>
            <script>
                $(document).ready(function() {{
                    let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{
                        defaultcolors: $3Dmol.rasmolElementColors,
                        backgroundColor: "white"
                    }});
                    
                    let pdb_content = `{to_visualize_pdb}`;
                    viewer.addModel(pdb_content, "pdb");
                    
                    viewer.setStyle({{}}, {{
                        cartoon: {{
                            colorscheme: {{
                                prop: 'b',
                                map: {color_map}
                            }}
                        }}
                        {', stick: {}' if show_sidechains else ''}
                    }});
                    
                    viewer.zoomTo();
                    viewer.render();
                }});
            </script>
        </body>
        </html>
        """
        return html_content
    except Exception as e:
        st.error(f"PDB 시각화 생성 중 오류: {str(e)}")
        return None              

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
        
def display_protein_structures():
    """단백질 구조 시각화 탭"""
    st.markdown("## 🔬 단백질 구조 시각화")
    try:
        # protein/pdb 디렉토리 존재 확인 및 생성
        pdb_dir = Path('protein/predictions')
        if not pdb_dir.exists():
            pdb_dir.mkdir(parents=True)
            st.warning("디렉토리가 비어있습니다.")
            return

        # 디렉토리 목록 가져오기
        directories = [d for d in pdb_dir.iterdir() if d.is_dir()]
        if not directories:
            st.warning("시각화할 수 있는 단백질 구조가 없습니다.")
            return
        # 컨트롤 패널
        selected_dir = st.selectbox(
            "단백질 구조 선택",
            options=[d.name for d in directories],
            format_func=lambda x: f"구조 ID: {x}",
            key="protein_structure_selector"
        )
        # 옵션
        show_sidechains = st.checkbox("측쇄 표시", value=False)
        
        if selected_dir:
            st.markdown(f"### {selected_dir} 구조 시각화")
            
            pdb_path = pdb_dir / selected_dir / "ranked_0.pdb"
            if not pdb_path.exists():
                st.error(f"PDB 파일을 찾을 수 없습니다: {pdb_path}")
                return
                
            html_content = visualization_pdb(str(pdb_path), show_sidechains)
            if html_content:
                # HTML을 안전하게 표시
                st.components.v1.html(html_content, height=620, scrolling=False)
                
    except Exception as e:
        st.error(f"구조 시각화 중 오류가 발생했습니다: {str(e)}")
        st.error("상세 에러 스택:")
        st.code(traceback.format_exc())

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
               - FASTA 파일을 업로드하여 AlphaFold2(높은 정확도, MSA 사용) 또는 ESMFold(빠른 속도, MSA 미사용)로 구조 예측
               - 예시: _"첨부한 fasta 파일에 존재하는 단백질 서열의 구조를 예측해줘"_
            """)
        
        st.markdown("---")
        
        # 2. 학습 데이터 관리 섹션
        st.markdown("## 📚 학습 데이터 관리")
        with st.expander("저장된 실험 데이터", expanded=True):
            display_experiment_list()
        
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
        
        st.markdown("---")
        
        # 5. 단백질 구조 시각화 섹션
        st.markdown("## 🔬 단백질 구조 시각화")
        if st.checkbox("구조 시각화 표시", value=False):
            display_protein_structures()

        # 파일 업로드 상태 관리 및 처리
        for upload_type, upload_file in [
            ('train', uploaded_train),
            ('test', uploaded_test),
            ('fasta', uploaded_fasta)
        ]:
            if upload_file is not None and upload_file != st.session_state.file_upload_state[upload_type]:
                with st.spinner(f'{upload_type.title()} 파일 처리 중...'):
                    handle_file_upload(upload_file, upload_type)
                st.session_state.file_upload_state[upload_type] = upload_file

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

            # AI 응답 생성 (스트리밍 방식)
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # 스트리밍 응답을 위한 처리
                for response_chunk in st.session_state.agent.chat_stream(prompt):
                    if response_chunk.get('current_response'):
                        chunk_text = response_chunk['current_response']
                        full_response += chunk_text
                        # 실시간으로 응답 업데이트
                        response_placeholder.markdown(full_response + "▌")
                
                # 최종 응답 표시
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # 데이터 분석 결과 시각화 (이전과 동일)
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