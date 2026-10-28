# Controle de acesso por reconhecimento facial

Sistema hipotetico de entrada em predio usando **OpenCV** e **DeepFace**.

## Estrutura

```
face_access_control/
  main.py                  ponto de entrada (CLI)
  requirements.txt
  src/
    config.py              caminhos, modelo, limiares, cores e textos
    image_io.py            leitura/escrita de imagens (OpenCV)
    face_engine.py         deteccao + embeddings (DeepFace)
    database.py            base de autorizados com cache de embeddings
    matcher.py             distancia cosseno e decisao liberado/negado
    visualizer.py          bounding boxes verde/vermelha (OpenCV)
    access_control.py      orquestra todo o fluxo
  data/
    authorized/            fotos das pessoas autorizadas
    test/                  fotos de teste
    output/                imagens anotadas e relatorio.csv
    cache/                 embeddings em cache
```

## Como usar

1. `pip install -r requirements.txt`
2. Coloque as fotos autorizadas em `data/authorized/`, uma subpasta por pessoa
   (ex.: `data/authorized/maria/foto1.jpg`, `data/authorized/maria/foto2.jpg`).
   Tambem funciona `data/authorized/maria.jpg`.
3. Coloque as fotos de teste em `data/test/`.
4. Execute `python main.py`.

Opcoes: `--model Facenet512|ArcFace|VGG-Face`, `--detector retinaface`, `--threshold 0.35`,
`--no-cache`, `--show`.

Verde com "acesso liberado" quando a distancia fica abaixo do limiar; vermelho com
"acesso negado" caso contrario. Na primeira execucao o DeepFace baixa os pesos do modelo.
