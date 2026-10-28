"""Ponto de entrada. Uso: python main.py [--model Facenet] [--show]"""
import argparse
from pathlib import Path

import cv2

from src import config
from src.access_control import AccessControlSystem
from src.image_io import list_images


def parse_args():
    p = argparse.ArgumentParser(description="Controle de acesso por reconhecimento facial")
    p.add_argument("--authorized", type=Path, default=config.AUTHORIZED_DIR)
    p.add_argument("--test", type=Path, default=config.TEST_DIR)
    p.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--model", default=config.MODEL_NAME, choices=list(config.COSINE_THRESHOLDS))
    p.add_argument("--detector", default=config.DETECTOR_BACKEND)
    p.add_argument("--threshold", type=float, default=None, help="limiar de distancia cosseno")
    p.add_argument("--no-cache", action="store_true", help="recalcula embeddings dos autorizados")
    p.add_argument("--show", action="store_true", help="exibe cada resultado em janela")
    return p.parse_args()


def main():
    args = parse_args()

    if not list_images(args.authorized):
        raise SystemExit(f"Nenhuma foto de autorizado encontrada em {args.authorized}")
    if not list_images(args.test):
        raise SystemExit(f"Nenhuma foto de teste encontrada em {args.test}")

    system = AccessControlSystem(
        model_name=args.model,
        detector_backend=args.detector,
        threshold=args.threshold,
        authorized_dir=args.authorized,
        use_cache=not args.no_cache,
    )
    pessoas = {e.person for e in system.entries}
    print(f"Autorizados carregados: {len(pessoas)} pessoa(s), {len(system.entries)} foto(s)")
    print(f"Modelo={args.model} | detector={args.detector} | limiar={system.threshold}\n")

    reports = system.process_folder(args.test, args.output)
    csv_path = system.write_report(reports, args.output)
    print(f"\nImagens anotadas e relatorio salvos em: {args.output}")
    print(f"Relatorio: {csv_path}")

    if args.show:
        for path in sorted(args.output.glob("*_resultado.jpg")):
            cv2.imshow(path.name, cv2.imread(str(path)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
