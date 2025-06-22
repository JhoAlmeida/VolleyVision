import cv2
import torch
import numpy as np
import torchvision.ops

def track_video(model, source, output_path, conf=0.5):
    try:
        from utils.torch_utils import select_device
        
        device = select_device('')
        
        # Configurações do vídeo
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception("Não foi possível abrir o vídeo")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Preparar saída
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Pré-processamento
            img = preprocess_frame(frame, device)
            
            # Inferência
            with torch.no_grad():
                if hasattr(model, 'predict'):  # Para modelos Ultralytics YOLO
                    results = model.predict(img, conf=conf)
                    pred = results[0].boxes.data
                else:  # Para modelos PyTorch padrão
                    pred = model(img)[0]
                    pred = non_max_suppression(pred, conf)
            
            # Desenhar resultados
            frame = draw_detections(frame, pred)
            out.write(frame)
            
        cap.release()
        out.release()
        return output_path
        
    except Exception as e:
        raise Exception(f"Erro no rastreamento: {str(e)}")

def preprocess_frame(frame, device):
    """Pré-processa o frame para inferência"""
    img = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    return img.unsqueeze(0)

def non_max_suppression(prediction, conf_thres=0.25):
    """Implementação simplificada de NMS"""
    if prediction.shape[1] != 6:  # [x1, y1, x2, y2, conf, cls]
        return prediction
        
    # Filtra por confiança
    mask = prediction[..., 4] > conf_thres
    prediction = prediction[mask]
    
    if prediction.shape[0] == 0:
        return None
    
    # NMS do torchvision
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    keep = torchvision.ops.nms(boxes, scores, 0.45)
    return prediction[keep]

def draw_detections(frame, detections):
    """Desenha as detecções no frame"""
    if detections is not None:
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = map(float, det[:6])
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{int(cls)}: {conf:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0,255,0), 1)
    return frame