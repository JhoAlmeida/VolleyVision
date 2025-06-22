import streamlit as st
import os
import torch
from utils.video_utils import download_youtube_video
from utils.tracking_utils import track_video
import time

# Configurações da página
st.set_page_config(
    page_title="Tracker de Vôlei",
    page_icon="🏐",
    layout="wide"
)

# Título do aplicativo
st.title("🏐 Rastreamento de Jogadores de Vôlei")

# Sidebar
with st.sidebar:
    st.header("Configurações")
    youtube_url = st.text_input("URL do YouTube", "")
    confidence = st.slider("Limite de Confiança", 0.1, 0.9, 0.5, 0.01)
    
    st.markdown("---")
    st.markdown("""
    ### Como usar:
    1. Cole a URL de um vídeo de vôlei
    2. Ajuste o limite de confiança
    3. Clique em Executar Rastreamento
    """)

# Função para carregar modelo
@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Verifica se o modelo existe
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            st.error(f"Modelo não encontrado em {model_path}")
            return None
            
        # Tenta carregar como PyTorch padrão
        try:
            model = torch.load(model_path, map_location=device)
            if isinstance(model, dict):
                raise RuntimeError("Arquivo contém apenas pesos, não o modelo completo")
            model.eval()
            st.success("Modelo carregado com sucesso!")
            return model
        except:
            # Tenta carregar como YOLO se falhar
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                st.success("Modelo YOLO carregado!")
                return model
            except:
                raise RuntimeError("Formato do modelo não reconhecido")
                
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

# Carrega o modelo
model = load_model()

# Processamento principal
if youtube_url and model:
    try:
        st.video(youtube_url)
        
        if st.button("Executar Rastreamento", type="primary"):
            with st.spinner("Processando (pode levar alguns minutos)..."):
                start_time = time.time()
                
                # Download do vídeo
                try:
                    video_path = download_youtube_video(youtube_url)
                    if not video_path:
                        raise Exception("Falha no download do vídeo")
                except Exception as e:
                    st.error(f"Erro no download: {str(e)}")
                    st.stop()
                
                # Rastreamento
                output_path = "output/resultado.mp4"
                os.makedirs("output", exist_ok=True)
                
                try:
                    track_video(
                        model=model,
                        source=video_path,
                        output_path=output_path,
                        conf=confidence
                    )
                except Exception as e:
                    st.error(f"Erro no rastreamento: {str(e)}")
                    st.stop()
                
                # Exibe resultados
                st.success(f"Processamento concluído em {time.time()-start_time:.2f} segundos!")
                st.video(output_path)
                
                # Botão de download
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Baixar Vídeo Processado",
                        f.read(),
                        file_name="rastreamento_volei.mp4",
                        mime="video/mp4"
                    )
    
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")

elif not model:
    st.warning("Modelo não carregado corretamente")
else:
    st.info("Cole uma URL do YouTube na barra lateral para começar")