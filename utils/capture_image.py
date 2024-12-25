import cv2


def capture_image_from_webcam(output_path='captured_face.jpg'):
    """
    Captura uma imagem da webcam e salva no caminho especificado.

    Args:
    output_path (str): Caminho onde a imagem capturada será salva.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    print("Pressione 'Espaço' para capturar a imagem e 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem.")
            break

        cv2.imshow('Capture Image', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Pressione 'Espaço' para capturar a imagem
            cv2.imwrite(output_path, frame)
            print(f"Imagem capturada e salva em {output_path}")
            break
        elif key == ord('q'):  # Pressione 'q' para sair sem capturar
            print("Saindo sem capturar a imagem.")
            break

    cap.release()
    cv2.destroyAllWindows()


# Exemplo de uso
if __name__ == "__main__":
    capture_image_from_webcam('captured_face.jpg')