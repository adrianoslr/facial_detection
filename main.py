import cv2
from utils.detection import detect_faces
from utils.recognition import FaceRecognizer
from utils.capture_image import capture_image_from_webcam  # Importa a função de captura de imagem


def main():
    recognizer = FaceRecognizer()

    while True:
        print("\n--- Sistema de Reconhecimento Facial ---")
        print("1. Adicionar novo rosto")
        print("2. Reconhecimento em tempo real")
        print("3. Capturar imagem da webcam")
        print("4. Sair")
        choice = input("Escolha uma opção: ")

        if choice == '1':
            name = input("Digite o nome da pessoa: ")
            capture_image_from_webcam('captured_face.jpg')  # Captura a imagem da webcam
            image = cv2.imread('captured_face.jpg')
            faces, _ = detect_faces(image)
            for face in faces:
                recognizer.add_face(name, face)
            print(f"Rosto de {name} registrado com sucesso!")
        elif choice == '2':
            print("Iniciando reconhecimento em tempo real...")
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                faces, boxes = detect_faces(frame)
                for face, box in zip(faces, boxes):
                    name = recognizer.recognize(face)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        elif choice == '3':
            capture_image_from_webcam('captured_image.jpg')  # Captura e salva uma imagem da webcam
        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")


if __name__ == "__main__":
    main()