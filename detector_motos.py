import cv2
import os
import time
from ultralytics import YOLO

class DetectorMotosYOLOv8:
    def __init__(self, modelo='yolov8n.pt'):
        """
        Inicializa o detector usando o modelo YOLOv8
        """
        print("Inicializando YOLOv8...")
        self.model = YOLO(modelo)  # Pode ser yolov8n.pt, yolov8s.pt, etc.
        self.output_dir = "motos_detectadas"
        os.makedirs(self.output_dir, exist_ok=True)
        print("YOLOv8 carregado com sucesso!")

    def detectar_motos(self, frame):
        """
        Detecta motos no frame usando YOLOv8
        """
        resultados = self.model.predict(source=frame, classes=[3], conf=0.5, verbose=False)  # Classe 3 = "motorcycle" no COCO
        anotacoes = resultados[0].boxes

        contador = 0
        for box in anotacoes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianca = float(box.conf[0])
            label = f"Moto: {confianca:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            contador += 1

        cv2.putText(frame, f"Motos detectadas: {contador}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, contador

    def processar_video(self, video_path):
        """
        Processa um vídeo ou webcam para detectar motos
        """
        if video_path.isdigit():
            video_path = int(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = os.path.join(self.output_dir, "motos_detectadas.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        print("Processando vídeo... Pressione 'q' para sair.")
        frame_count = 0
        tempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro.")
                break

            frame_count += 1
            frame_processado, contador_motos = self.detectar_motos(frame)
            out.write(frame_processado)
            cv2.imshow("Detector de Motos (YOLOv8)", frame_processado)

            # Salvar imagem a cada 30 frames se houver detecção
            if frame_count % 30 == 0 and contador_motos > 0:
                timestamp = int(time.time())
                img_filename = os.path.join(self.output_dir, f"moto_detectada_{timestamp}.jpg")
                cv2.imwrite(img_filename, frame_processado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        tempo_total = time.time() - tempo_inicio
        print(f"Processamento concluído em {tempo_total:.2f} segundos.")
        print(f"Vídeo salvo em: {output_filename}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    print("==========================================")
    print(" DETECTOR DE MOTOS COM YOLOv8")
    print("==========================================")
    detector = DetectorMotosYOLOv8(modelo='yolov8n.pt')  # Você pode usar yolov8s.pt, yolov8m.pt etc.

    video_path = input("Digite o caminho do vídeo ou número da câmera (0 para webcam): ")
    detector.processar_video(video_path)

if __name__ == "__main__":
    main()
