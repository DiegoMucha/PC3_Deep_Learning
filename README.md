# Semana 6 · CNN para clasificación de células sanguíneas

Implementación en **PyTorch** de `PC2 - 2026 II-1.pdf`: LeNet-5 y VGG-11 adaptadas,
Batch Normalization (BN) y transferencia con ResNet-18. Autor: **[nombre y apellidos]**.

**Estado:** código preparado; notebooks y entrenamientos **sin ejecutar**, por indicación
del autor. Las métricas y conclusiones quedan pendientes de ejecutar en Colab. El informe
se redactará aparte. Para entregar faltan los notebooks con salidas, los resultados,
`informe.pdf` y publicar el repositorio como `dl-semana6-[apellidos]`.

## Organización

```text
├── README.md
├── requirements.txt
├── data/
│   ├── download_data.sh
│   └── raw/images/              # Dataset descargado localmente
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_lenet_vgg.ipynb
│   └── 03_transfer.ipynb
├── src/
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── results/
│   └── figures/
└── informe.pdf                 # Pendiente de redactar por el autor
```

El dataset ya está en `data/raw/`. Al ejecutar se crean `data/splits/` y `results/runs/`. Se versionan tablas,
figuras y notebooks ejecutados, no dataset, credenciales, entornos ni checkpoints grandes.
El informe final se añadirá a la raíz.

## Instalación local con uv

```bash
# El entorno local .venv ya está instalado.
source .venv/bin/activate
python -m jupyterlab
```

Para crear el entorno en otro equipo: `uv venv --python 3.12` y
`uv pip install -r requirements.txt`. También puedes usar Python ≥3.10,
con una distribución de PyTorch compatible con su CPU/CUDA. Se recomienda GPU para
entrenar. El entorno instalado se conserva; ahora se usa `uv pip`, sin `uv sync`.

## Google Colab

1. Sube la carpeta `PC3_Deep_Learning` a **Mi unidad** de Google Drive. Incluye
   `data/raw/` con las imágenes descargadas; no subas `.venv` ni `.git`.
2. Abre los notebooks en Colab y activa GPU para los notebooks 02 y 03.
3. Ejecuta **01 → 02 → 03**. Los tres usan la ruta fija
   `/content/drive/MyDrive/PC3_Deep_Learning`. El 01 prepara las particiones;
   el 02 y el 03 las leen directamente.
4. Resultados, checkpoints y README actualizado se guardan en esa misma carpeta de Drive.
5. Guarda los notebooks con sus salidas. El notebook 03 descarga `resultados_colab.zip`
   con tablas, figuras, métricas, particiones y README, sin imágenes ni checkpoints.
6. Copia los resultados al repositorio y redacta los hallazgos. Conserva `best.pt` para la demo.

Las rutas están escritas directamente en las celdas, sin detectar carpetas ni cambiar
automáticamente entre Colab y local. Los notebooks están preparados para Colab;
para trabajar localmente se pueden usar los comandos de terminal indicados abajo.
Los hiperparámetros siguen en `src/train.py`. No hace falta descargar nuevamente los
datos en Colab: se usan los que subiste a Drive.

## Datos y metodología

[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells):
EOSINOPHIL, LYMPHOCYTE, MONOCYTE y NEUTROPHIL. El enunciado indica aproximadamente
12 500 imágenes RGB de 320×240 y licencia CC0; el EDA mide conteos y tamaños reales.

La descarga local corresponde a la versión 6 de Kaggle. Se verificó la lectura de
**12 444 imágenes** de TRAIN/TEST: **9 957** de entrenamiento y **2 487** de test,
todas RGB de **320×240**. Los datos están disponibles localmente y excluidos de Git.

Se conserva TRAIN/TEST oficial, sin usar TEST_SIMPLE. Se reserva 20% de TRAIN para
validación estratificando grupos de hashes de píxeles idénticos; la proporción por imágenes
puede variar ligeramente. `manifest.csv` guarda rutas, clases, dimensiones, hashes y split.
Los duplicados TRAIN/TEST se notifican conservando el split solicitado. Los hashes no
detectan aumentos de una misma célula ni pacientes: discutir esta limitación de independencia.

Train: flips horizontal/vertical y rotación ±15°. Val/test: sin aumento. CNN: RGB 64×64,
normalización fija media/desviación 0.5. ResNet: RGB 224×224, normalización ImageNet.
Se redimensiona la imagen completa, en lugar del recorte central de ImageNet; esto
deforma la relación de aspecto. No se estiman estadísticas usando test.

| Familia | Variantes | Optimizador | LR | Épocas |
| --- | --- | --- | --- | --- |
| Desde cero | LeNet, LeNet+BN, VGG-11, VGG-11+BN | SGD, momentum 0.9 | 0.01 | 10 |
| Control LR alto | LeNet y LeNet+BN | SGD, momentum 0.9 | 0.1 | 10 |
| ResNet-18 | Solo FC | Adam | 0.001 | 5 |
| ResNet-18 | layer3, layer4 y FC | Adam | 0.0001 | 5 |
| ResNet-18 | Todas las capas | Adam | 0.00001 | 5 |

Total: **9 experimentos y 75 épocas** (4×10 + 2×10 + 3×5).

Común: semilla 42, batch 64, weight decay 0.0001. Configuración en
`DEFAULT_SETTINGS` de `src/train.py`. LeNet conserva dos conv-pool y tres FC, tanh y average pool.
VGG conserva ocho conv y tres FC, mitad de canales y cabeza 1024→512→256→4 para ahorrar
memoria. BN se inserta después de cada conv. El bucle de entrenamiento es propio.

Para los pares ±BN se mantienen pesos iniciales compartidos, split, orden de lotes,
aumentos y presupuesto. Sin scheduler, early stopping ni clipping. La convergencia se
define como primera época con **accuracy de validación ≥0.80**; vacío significa no
alcanzado. LR alta se prueba solo en LeNet con/sin BN. `unstable` indica
NaN/Inf; las oscilaciones finitas se estudian en las curvas.

En transferencia también se congelan las estadísticas BN. Ajuste parcial congela stem,
layer1 y layer2. Se usan pesos ImageNet `IMAGENET1K_V1`. La comparación con CNN desde
cero también cambia resolución y optimizador; no aísla solo preentrenamiento.

La mejor época y el mejor modelo se seleccionan por accuracy de validación, pérdida
como desempate. Test se evalúa al final de cada corrida. La selección final excluye
los controles de LR alta. No ajustar hiperparámetros a partir de test. Una semilla
no cuantifica incertidumbre: se pueden repetir semillas de entrenamiento manteniendo
fijo el manifest.

## Terminal (opcional: estos comandos sí entrenan)

Desde la raíz, con el entorno activo:

```bash
bash data/download_data.sh
python -m src.utils prepare
python -m src.train --group scratch
python -m src.train --group high_lr
python -m src.train --group transfer
python -m src.train --export-only --update-readme
```

`--group all` ejecuta los nueve experimentos. `--device cpu/cuda` fija dispositivo;
por defecto es automático. `--output` permite otra carpeta. No se reutilizan resultados
si cambia código, configuración o manifest. Las corridas completadas se omiten; una
corrida interrumpida **reinicia desde su semilla**, sin reanudación exacta a mitad.
`best.pt` guarda el mejor modelo y `last.pt` el último estado, incluido optimizador.

Cada corrida exporta historia CSV, métricas JSON, predicciones de test, reporte por
clase y entorno. El tiempo medio por época incluye lectura de datos y entrenamiento,
excluye validación, figuras, descarga y guardado. Se sincroniza CUDA al medir.
Comparar tiempos sobre el mismo hardware.

## Resultados

<!-- RESULTS:START -->

Experimentos completados: 6/6 registrados. Accuracy en escala 0–1; — en epoch_to_80_val significa que no llegó al 80%. La selección usa validación; los hallazgos se redactan tras revisar las curvas.

| name | parameters | trainable_parameters | mean_train_seconds | best_val_accuracy | test_accuracy | test_f1_macro | epoch_to_80_val | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lenet_bn_scratch_seed42 | 338020 | 338020 | 19.3199 | 0.5868 | 0.3989 | 0.4038 | — | completed |
| lenet_scratch_seed42 | 337976 | 337976 | 18.9143 | 0.7129 | 0.4865 | 0.4883 | — | completed |
| resnet18_feature_seed42 | 11178564 | 2052 | 230.3541 | 0.5226 | 0.4085 | 0.3773 | — | completed |
| resnet18_full_seed42 | 11178564 | 11178564 | 674.7028 | 0.9910 | 0.8532 | 0.8557 | 2.0000 | completed |
| resnet18_partial_seed42 | 11178564 | 10495492 | 381.3263 | 0.9990 | 0.8058 | 0.8050 | 1.0000 | completed |
| vgg11_scratch_seed42 | 2963396 | 2963396 | 65.7415 | 0.2510 | 0.2509 | 0.1003 | — | completed |

<!-- RESULTS:END -->

**Hallazgos por completar:** modelo elegido por validación, efecto de BN sobre
convergencia/LR alta, estrategia recomendada y errores por clase. Usar `results/tables/`
y `results/figures/`; no afirmar mejoras antes de medirlas.

## Informe y referencias

La plantilla está al final de este README para copiarla a Word, Google Docs o tu editor.
Redacta el informe aparte, con un máximo de ocho páginas sin referencias, y exporta
`informe.pdf` en la raíz. Las referencias originales son:

- LeCun et al. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791
- Krizhevsky et al. (2012). *ImageNet classification with deep convolutional neural networks*. NeurIPS 25. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- Simonyan y Zisserman (2014/2015). *Very deep convolutional networks for large-scale image recognition*. Preprint 2014; ICLR 2015. https://arxiv.org/abs/1409.1556
- He et al. (2016). *Deep residual learning for image recognition*. CVPR, 770–778. https://doi.org/10.1109/CVPR.2016.90
- Ioffe y Szegedy (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift*. ICML, 448–456. https://proceedings.mlr.press/v37/ioffe15.html

Pesos y normalización: [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).


## Plantilla del informe

**Título:** Arquitecturas CNN para la clasificación de células sanguíneas

**Autor y código:** [completar] · **Curso/docente:** [completar]
**Fecha:** [completar] · **Repositorio público:** [URL]

> Sustituir campos e instrucciones por redacción propia. Máximo **8 páginas de contenido**,
> sin referencias. Exportar `informe.pdf` en la raíz. Distribución sugerida: introducción
> 0.75 pág.; metodología 2; resultados 2.5; discusión 1.5; conclusiones 0.5.
> Reservar espacio para título y figuras. No presentar hipótesis como resultados medidos.

### 1. Introducción

[Motivar la clasificación de células sanguíneas, presentar objetivo y preguntas sobre
arquitectura, BN y transferencia. Describir fuente, clases, cantidades reales, dimensiones
y licencia del dataset. Distinguir experimento académico y validación clínica.]

**Objetivo general:** [completar]. **Preguntas experimentales:** [completar].

#### 2. Metodología

#### 2.1 Datos y preprocesamiento

| Partición | EOSINOPHIL | LYMPHOCYTE | MONOCYTE | NEUTROPHIL | Total |
| --- | --- | --- | --- | --- | --- |
| Train | [ ] | [ ] | [ ] | [ ] | [ ] |
| Validación | [ ] | [ ] | [ ] | [ ] | [ ] |
| Test | [ ] | [ ] | [ ] | [ ] | [ ] |

Fuente: `tables/class_distribution.csv`. [Explicar 20% de TRAIN para validación,
estratificación por grupos de duplicados, semilla, TEST oficial y exclusión de TEST_SIMPLE.
Documentar `data_audit/audit.json` y limitaciones por paciente/aumentos preexistentes.]

[RGB, resize 64×64/224×224, normalización, aumentos solo en train; justificar sus
decisiones y comentar deformación de relación de aspecto.]

#### 2.2 Arquitecturas

[Explicar LeNet y adaptación respecto a [1]; VGG-A simplificada respecto a [3].
AlexNet [2] es antecedente histórico, no modelo de estos experimentos.]

| Modelo | Entrada | Canales conv | FC | Parámetros |
| --- | --- | --- | --- | --- |
| LeNet | 64×64×3 | 6,16 | 120,84,4 | [ ] |
| LeNet+BN | 64×64×3 | 6,16 | 120,84,4 | [ ] |
| VGG-11 | 64×64×3 | 32,64,128,128,256,256,256,256 | 512,256,4 | [ ] |
| VGG-11+BN | 64×64×3 | Igual | Igual | [ ] |

#### 2.3 Protocolo y análisis BN

[Copiar hiperparámetros reales: semilla, batch, épocas, optimizador, LR, decay;
hardware y versiones de `runs/*/environment.json`. Documentar cambios de configuración.]

[Controles BN: split, inicialización compartida, lotes y aumentos iguales; LR 0.01 y 0.1.
Definir primera época con accuracy val ≥80%; «no alcanzado» cuando corresponda.
Selección por accuracy val/pérdida; test solo para evaluación final. Definir métricas
y tiempo por época excluyendo validación/guardado.]

#### 2.4 Transfer Learning

[ResNet-18 preentrenada [4]. Explicar las tres estrategias, congelamiento de stem y
dos primeros bloques, estadísticas BN y LR 1e−3/1e−4/1e−5.]

#### 3. Resultados

#### 3.1 Cuatro CNN desde cero

| Modelo | Parámetros | s/época | Accuracy val | Accuracy test | Macro-F1 test |
| --- | --- | --- | --- | --- | --- |
| LeNet | [ ] | [ ] | [ ] | [ ] | [ ] |
| LeNet+BN | [ ] | [ ] | [ ] | [ ] | [ ] |
| VGG-11 | [ ] | [ ] | [ ] | [ ] | [ ] |
| VGG-11+BN | [ ] | [ ] | [ ] | [ ] | [ ] |

Fuente: `tables/scratch.csv`. Aclarar si accuracy es proporción o porcentaje.

**Figura 1:** [curvas pérdida/accuracy; ejes, título y leyenda].
[Describir patrones observados y brechas train/val con números concretos.]

#### 3.2 BN y tasa de aprendizaje

| Arquitectura | BN | LR | Época ≥80% val | Mejor accuracy val | Estabilidad |
| --- | --- | --- | --- | --- | --- |
| LeNet | No | 0.01 | [ ] | [ ] | [ ] |
| LeNet | Sí | 0.01 | [ ] | [ ] | [ ] |
| LeNet | No | 0.1 | [ ] | [ ] | [ ] |
| LeNet | Sí | 0.1 | [ ] | [ ] | [ ] |
| VGG | No | 0.01 | [ ] | [ ] | [ ] |
| VGG | Sí | 0.01 | [ ] | [ ] | [ ] |

Fuentes: `scratch.csv`, `high_lr.csv`. Distinguir NaN/Inf, oscilación finita y falta
de convergencia. **Figura 2:** [curvas train/val ±BN, LR base y alta].

#### 3.3 Transferencia

| Estrategia | Parámetros entrenables | s/época | Accuracy val | Accuracy test | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Mejor CNN desde cero: [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Feature extraction | [ ] | [ ] | [ ] | [ ] | [ ] |
| Fine-tuning parcial | [ ] | [ ] | [ ] | [ ] | [ ] |
| Fine-tuning total | [ ] | [ ] | [ ] | [ ] | [ ] |

Fuente: `tables/transfer_vs_best_scratch.csv`. Selección por validación.

**Figura 3:** [matriz de confusión en test del mejor modelo según validación].
[Clases confundidas y recall de `classification_report.json`.]

### 4. Discusión

**BN e Internal Covariate Shift:** [Explicar motivación original de [5], normalización
por minibatch, parámetros aprendibles y estadísticas para evaluación. Relacionar con
observaciones sin afirmar que las curvas miden directamente ICS o prueban su mecanismo.]

**Recomendación con datos limitados:** [Escoger con evidencia de generalización,
recall por clase, capacidad, coste y sobreajuste; indicar cuándo cambiaría la elección.]

**Limitaciones:** [semillas, aumentos correlacionados, duplicados, falta de grupos por
paciente, diferencias de resolución/optimizador/preentrenamiento, validación externa].

**Mejoras:** [priorizar dos o tres propuestas y justificar].

### 5. Conclusiones

1. [Conclusión cuantitativa sobre arquitecturas.]
2. [Efecto medido de BN y LR alta.]
3. [Recomendación de transferencia y límite principal de evidencia.]

### Referencias IEEE (fuera del límite)

[1] Y. LeCun, L. Bottou, Y. Bengio y P. Haffner, “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998, doi: 10.1109/5.726791.

[2] A. Krizhevsky, I. Sutskever y G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” *Advances in Neural Information Processing Systems*, vol. 25, 2012.

[3] K. Simonyan y A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” *ICLR*, 2015. Preprint arXiv:1409.1556, 2014.

[4] K. He, X. Zhang, S. Ren y J. Sun, “Deep residual learning for image recognition,” *CVPR*, pp. 770–778, 2016, doi: 10.1109/CVPR.2016.90.

[5] S. Ioffe y C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” *ICML*, PMLR 37, pp. 448–456, 2015.

[6] P. T. Mooney, “Blood Cell Images,” Kaggle. https://www.kaggle.com/datasets/paultimothymooney/blood-cells (consulta: [fecha]).
