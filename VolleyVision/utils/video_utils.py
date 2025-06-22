import os
import logging
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, RegexMatchError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_youtube_video(url, output_path="temp"):
    try:
        # Corrige URLs encurtadas
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        yt = YouTube(url)
        
        # Pega a stream com melhor qualidade
        stream = yt.streams.filter(
            file_extension='mp4',
            progressive=True
        ).order_by('resolution').desc().first()
        
        if not stream:
            raise Exception("Nenhuma stream MP4 disponível")
        
        # Cria diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Define caminho do arquivo
        filename = f"video_{yt.video_id}.mp4"
        video_path = os.path.join(output_path, filename)
        
        # Download
        stream.download(output_path=output_path, filename=filename)
        
        return video_path
        
    except VideoUnavailable:
        raise Exception("Vídeo indisponível (privado/removido)")
    except RegexMatchError:
        raise Exception("URL do YouTube inválida")
    except Exception as e:
        raise Exception(f"Erro no download: {str(e)}")