# Curso de NLP de Hugging Face 🤗

**Parte 1: Fundamentos de Transformadores y NLP**

Este curso te enseñará los fundamentos del procesamiento de lenguaje natural (NLP) usando las bibliotecas del ecosistema Hugging Face: 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers y 🤗 Accelerate.

---

# 0. Configuración

# Introducción

Bienvenido al curso de Hugging Face. Esta introducción te guiará en la configuración de un entorno de trabajo. Si acabas de empezar el curso, te recomendamos que primero eches un vistazo al [Capítulo 1](/course/chapter1), y luego vuelvas y configures tu entorno para poder probar el código por ti mismo.

Todas las librerías que usaremos en este curso están disponibles como paquetes de Python, así que aquí te mostraremos cómo configurar un entorno de Python e instalar las librerías específicas que necesitarás.

Cubriremos dos formas de configurar tu entorno de trabajo, utilizando un cuaderno Colab o un entorno virtual Python. Siéntete libre de elegir la que más te convenga. Para los principiantes, recomendamos encarecidamente que comiencen utilizando un cuaderno Colab.

Tenga en cuenta que no vamos a cubrir el sistema Windows. Si está utilizando Windows, le recomendamos que siga utilizando un cuaderno Colab. Si está utilizando una distribución de Linux o macOS, puede utilizar cualquiera de los enfoques descritos aquí.

La mayor parte del curso depende de que tengas una cuenta de Hugging Face. Te recomendamos que crees una ahora: [crear una cuenta](https://huggingface.co/join).

## Uso de un cuaderno Google Colab

Utilizar un cuaderno Colab es la configuración más sencilla posible; ¡arranca un cuaderno en tu navegador y ponte a codificar directamente! 

Si no estás familiarizado con Colab, te recomendamos que empieces siguiendo la [introducción](https://colab.research.google.com/notebooks/intro.ipynb). Colab te permite utilizar algún hardware de aceleración, como GPUs o TPUs, y es gratuito para cargas de trabajo pequeñas.

Una vez que te sientas cómodo moviéndote en Colab, crea un nuevo notebook y comienza con la configuración:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter0/new_colab.png" alt="An empty colab notebook" width="80%"/>
</div>

El siguiente paso es instalar las librerías que usaremos en este curso. Usaremos `pip` para la instalación, que es el gestor de paquetes para Python. En los cuadernos, puedes ejecutar comandos del sistema precediéndolos con el carácter `!`, así que puedes instalar la librería 🤗 Transformers de la siguiente manera:

```
!pip install transformers
```

Puede asegurarse de que el paquete se ha instalado correctamente importándolo en su tiempo de ejecución de Python:

```
import transformers
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter0/install.gif" alt="A gif showing the result of the two commands above: installation and import" width="80%"/>
</div>

Esto instala una versión muy ligera de 🤗 Transformers. En particular, no se instalan frameworks específicos de deep learning (como PyTorch o TensorFlow). Dado que vamos a utilizar un montón de características diferentes de la librería, se recomienda instalar la versión de desarrollo, que viene con todas las dependencias necesarias para casi cualquier caso de uso imaginable:

```
!pip install transformers[sentencepiece]
```

Esto te llevará un poco de tiempo, pero luego estarás listo para el resto del curso.

## Usar un entorno virtual de Python

Si prefieres utilizar un entorno virtual de Python, el primer paso es instalar Python en tu sistema. Recomendamos seguir [esta guía](https://realpython.com/installing-python/) para empezar.

Una vez que tengas Python instalado, deberías poder ejecutar comandos de Python en tu terminal. Puedes empezar ejecutando el siguiente comando para asegurarte de que está correctamente instalado antes de proceder a los siguientes pasos: `python --version`. Esto debería imprimir la versión de Python disponible en tu sistema.

Cuando ejecutes un comando de Python en tu terminal, como `python --version`, debes pensar en el programa que ejecuta tu comando como el Python "principal" de tu sistema. Recomendamos mantener esta instalación principal libre de paquetes, y usarla para crear entornos separados para cada aplicación en la que trabajes - de esta manera, cada aplicación puede tener sus propias dependencias y paquetes, y no tendrás que preocuparte por posibles problemas de compatibilidad con otras aplicaciones.

En Python esto se hace con [*entornos virtuales*](https://docs.python.org/3/tutorial/venv.html), que son árboles de directorios autocontenidos que contienen cada uno una instalación de Python con una versión particular de Python junto con todos los paquetes que la aplicación necesita. La creación de un entorno virtual de este tipo puede hacerse con varias herramientas diferentes, pero nosotros utilizaremos el paquete oficial de Python para este fin, que se llama [`venv`](https://docs.python.org/3/library/venv.html#module-venv).

En primer lugar, crea el directorio en el que te gustaría que viviera tu aplicación - por ejemplo, podrías crear un nuevo directorio llamado *transformers-course* en la raíz de tu directorio personal:

```
mkdir ~/transformers-course
cd ~/transformers-course
```

Desde este directorio, crea un entorno virtual utilizando el módulo `venv` de Python:

```
python -m venv .env
```

Ahora debería tener un directorio llamado *.env* en su carpeta, por lo demás vacía:

```
ls -a
```

```out
.      ..    .env
```

Puedes entrar y salir de tu entorno virtual con los scripts `activate` y `deactivate`:

```
# Activa el entorno virtual
source .env/bin/activate

# Desactiva el entorno virtual
deactivate
```

Puedes asegurarte de que el entorno está activado ejecutando el comando `which python`: si apunta al entorno virtual, entonces lo has activado con éxito.

```
which python
```

```out
/home/<user>/transformers-course/.env/bin/python
```

### Instalación de dependencias

Al igual que en la sección anterior sobre el uso de las instancias de Google Colab, ahora necesitarás instalar los paquetes necesarios para continuar. De nuevo, puedes instalar la versión de desarrollo de 🤗 Transformers utilizando el gestor de paquetes `pip`:

```
pip install "transformers[sentencepiece]"
```

Ya está todo preparado y listo para funcionar.


---

# 1. Modelos de Transformadores

# Introducción


## ¡Te damos la bienvenida al curso de 🤗!


**Video:** [Ver en YouTube](https://youtu.be/00GKzGyWFEs)


Este curso te enseñará sobre procesamiento de lenguaje natural (PLN) usando librerías del ecosistema [Hugging Face](https://huggingface.co/) - [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Tokenizers](https://github.com/huggingface/tokenizers) y [🤗 Accelerate](https://github.com/huggingface/accelerate) — así como el [Hub de Hugging Face](https://huggingface.co/models). El curso es completamente gratuito y sin anuncios.

## ¿Qué esperar?

Esta es una pequeña descripción del curso:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/summary.svg" alt="Brief overview of the chapters of the course.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/summary-dark.svg" alt="Brief overview of the chapters of the course.">
</div>

- Los capítulos 1 a 4 ofrecen una introducción a los conceptos principales de la librería 🤗 Transformers. Al final de esta sección del curso, estarás familiarizado con la manera en que trabajan los Transformadores y sabrás cómo usar un modelo del [Hub de Hugging Face](https://huggingface.co/models), ajustarlo a tu conjunto de datos y compartir tus resultados en el Hub.
- Los capítulos 5 a 8 enseñan lo básico de 🤗 Datasets y 🤗 Tokenizers antes de entrar en tareas clásicas de PLN. Al final de esta sección, podrás abordar por ti mismo los problemas más comunes de PLN.
- Los capítulos 9 al 12 van más allá del PLN y exploran cómo los Transformadores pueden abordar tareas de procesamiento del habla y visión por computador. A lo largo del camino, aprenderás a construir y compartir demos de tus modelos, así como optimizarlos para entornos de producción. Al final de esta sección, estarás listo para aplicar 🤗 Transformers a (casi) cualquier problema de Machine Learning.

Este curso:

* Requiere amplio conocimiento de Python
* Debería ser tomado después de un curso de introducción a deep learning, como [Practical Deep Learning for Coders](https://course.fast.ai/) de [fast.ai's](https://www.fast.ai/) o alguno de los programas desarrollados por [DeepLearning.AI](https://www.deeplearning.ai/)
* No necesita conocimiento previo de [PyTorch](https://pytorch.org/) o [TensorFlow](https://www.tensorflow.org/), aunque un nivel de familiaridad con alguno de los dos podría ser útil

Después de que hayas completado este curso, te recomendamos revisar la [Especialización en Procesamiento de Lenguaje Natural](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearning-ai&utm_medium=institutions&utm_campaign=20211011-PLN-2-hugging_face-page-PLN-refresh) de DeepLearning.AI, que cubre un gran número de modelos tradicionales de PLN como Naive Bayes y LSTMs.

## ¿Quiénes somos?

Acerca de los autores:

**Matthew Carrigan** es Ingeniero de Machine Learning en Hugging Face. Vive en Dublin, Irlanda y anteriormente trabajó como Ingeniero ML en Parse.ly y como investigador post-doctoral en Trinity College Dublin. No cree que vamos a alcanzar una Inteligencia Artificial General escalando arquitecturas existentes, pero en todo caso tiene grandes expectativas sobre la inmortalidad robótica.

**Lysandre Debut** es Ingeniero de Machine Learning en Hugging Face y ha trabajado en la librería 🤗 Transformers desde sus etapas de desarrollo más tempranas. Su objetivo es hacer que el PLN sea accesible para todos a través del desarrollo de herramientas con una API muy simple.

**Sylvain Gugger** es Ingeniero de Investigación en Hugging Face y uno de los principales mantenedores de la librería 🤗 Transformers. Anteriormente fue Científico de Investigación en fast.ai y escribió _[Deep Learning for Coders with fastai and PyTorch](https://learning.oreilly.com/library/view/deep-learning-for/9781492045519/)_ junto con Jeremy Howard. El foco principal de su investigación es hacer el deep learning más accesible, al diseñar y mejorar técnicas que permiten un entrenamiento rápido de modelos con recursos limitados.

**Merve Noyan** es Promotora de Desarrolladores en Hugging Face, trabaja en el desarrollo de herramientas y construcción de contenido relacionado, con el fín de democratizar el machine learning para todos.

**Lucile Saulnier** es Ingeniera de Machine Learning en Hugging Face, donde desarrolla y apoya el uso de herramientas de código abierto. Ella está activamente involucrada en varios proyectos de investigación en el campo del Procesamiento de Lenguaje Natural como entrenamiento colaborativo y BigScience.

**Lewis Tunstall**  es Ingeniero de Machine Learning en Hugging Face, enfocado en desarrollar herramientas de código abierto y hacerlas accesibles a la comunidad en general. También es coautor de un próximo [libro de O'Reilly sobre Transformadores](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).

**Leandro von Werra**  es Ingeniero de Machine Learning en el equipo de código abierto en Hugging Face y coautor de un próximo [libro de O'Reilly sobre Transformadores](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/). Tiene varios años de experiencia en la industria llevando modelos de PLN a producción, trabajando a lo largo de todo el entorno de Machine Learning. 

¿Estás listo para comenzar? En este capítulo vas a aprender:
* Cómo usar la función `pipeline()` para resolver tareas de PLN como la generación y clasificación de texto
* Sobre la arquitectura de los Transformadores
* Cómo distinguir entre las arquitecturas de codificador, decodificador y codificador-decofidicador, además de sus casos de uso

---

# Procesamiento de Lenguaje Natural


Antes de ver los Transformadores, hagamos una revisión rápida de qué es el procesamiento de lenguaje natural y por qué nos interesa.

## ¿Qué es PLN?

El PLN es un campo de la lingüística y el machine learning enfocado en entender todo lo relacionado con el lenguaje humano. El objetivo de las tareas de PLN no sólo es entender palabras de manera individual, sino también entender su contexto.

Esta es una lista de tareas comunes de PLN, con algunos ejemplos:

- **Clasificar oraciones enteras**: Obtener el sentimiento de una reseña, detectar si un correo electrónico es spam, determinar si una oración es gramaticalmente correcta o si dos oraciones están lógicamente relacionadas entre si
- **Clasificar cada palabra en una oración**: Identificar los componentes gramaticales de una oración (sustantivo, verbo, adjetivo) o las entidades nombradas (persona, ubicación, organización)
- **Generar texto**: Completar una indicación con texto generado automáticamente, completar los espacios en blanco en un texto con palabras ocultas
- **Extraer una respuesta de un texto**: Dada una pregunta y un contexto, extraer la respuesta a la pregunta en función de la información proporcionada en el contexto
- **Generar una nueva oración de un texto de entrada**: Traducir un texto a otro idioma, resumir un texto

No obstante, el PLN no se limita a texto escrito. También aborda desafíos complejos en reconocimiento del habla y visión por computador, como generar la transcripción de una muestra de audio o la descripción de una imagen.

## ¿Por qué es retador?

Los computadores no procesan la información en la misma forma que los humanos. Por ejemplo, cuando leemos la oración "tengo hambre", podemos identificar fácilmente su significado. De manera similar, dadas dos oraciones como "tengo hambre" y "estoy triste", podemos determinar fácilmente qué tan parecidas son. Para los modelos de Machine Learning (ML) estas tareas son más difíciles. El texto necesita ser procesado de tal forma que le permita al modelo aprender de él. Y como el lenguaje es complejo, necesitamos pensar con cuidado cómo debe ser este procesamiento. Hay gran cantidad de investigación sobre cómo representar texto y vamos a ver algunos métodos en el siguiente capítulo.


---

# Transformadores, ¿qué pueden hacer?


En esta sección, veremos qué pueden hacer los Transformadores y usaremos nuestra primera herramienta de la librería 🤗 Transformers: la función `pipeline()`.

> [!TIP]
> 👀 Ves el botón <em>Open in Colab</em> en la parte superior derecha? Haz clic en él para abrir un cuaderno de Google Colab con todos los ejemplos de código de esta sección. Este botón aparecerá en cualquier sección que tenga ejemplos de código.
>
> Si quieres ejecutar los ejemplos localmente, te recomendamos revisar la <a href="/course/chapter0">configuración</a>.

## ¡Los Transformadores están en todas partes!

Los Transformadores se usan para resolver todo tipo de tareas de PLN, como las mencionadas en la sección anterior. Aquí te mostramos algunas de las compañías y organizaciones que usan Hugging Face y Transformadores, que también contribuyen de vuelta a la comunidad al compartir sus modelos:

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/companies.PNG" alt="Companies using Hugging Face" width="100%">

La [librería 🤗 Transformers](https://github.com/huggingface/transformers) provee la funcionalidad de crear y usar estos modelos compartidos. El [Hub de Modelos](https://huggingface.co/models) contiene miles de modelos preentrenados que cualquiera puede descargar y usar. ¡Tú también puedes subir tus propios modelos al Hub!

> [!TIP]
> ⚠️ El Hub de Hugging Face no se limita a Transformadores. ¡Cualquiera puede compartir los tipos de modelos o conjuntos de datos que quiera! ¡<a href="https://huggingface.co/join">Crea una cuenta de huggingface.co</a> para beneficiarte de todas las funciones disponibles!

Antes de ver cómo funcionan internamente los Transformadores, veamos un par de ejemplos sobre cómo pueden ser usados para resolver tareas de PLN. 

## Trabajando con pipelines


**Video:** [Ver en YouTube](https://youtu.be/tiZFewofSLM)


El objeto más básico en la librería 🤗 Transformers es la función `pipeline()`. Esta función conecta un modelo con los pasos necesarios para su preprocesamiento y posprocesamiento, permitiéndonos introducir de manera directa cualquier texto y obtener una respuesta inteligible:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

```python out
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

¡Incluso podemos pasar varias oraciones!

```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
```

```python out
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

Por defecto, este pipeline selecciona un modelo particular preentrenado que ha sido ajustado para el análisis de sentimientos en Inglés. El modelo se descarga y se almacena en el caché cuando creas el objeto `classifier`. Si vuelves a ejecutar el comando, se usará el modelo almacenado en caché y no habrá necesidad de descargarlo de nuevo.

Hay tres pasos principales que ocurren cuando pasas un texto a un pipeline:

1. El texto es preprocesado en un formato que el modelo puede entender.
2. La entrada preprocesada se pasa al modelo.
3. Las predicciones del modelo son posprocesadas, de tal manera que las puedas entender.

Algunos de los [pipelines disponibles](https://huggingface.co/transformers/main_classes/pipelines.html) son:

- `feature-extraction` (obtener la representación vectorial de un texto)
- `fill-mask`
- `ner` (reconocimiento de entidades nombradas)
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

¡Veamos algunas de ellas!

## Clasificación *zero-shot*

Empezaremos abordando una tarea más compleja, en la que necesitamos clasificar textos que no han sido etiquetados. Este es un escenario común en proyectos de la vida real porque anotar texto usualmente requiere mucho tiempo y dominio del tema. Para este caso de uso, el pipeline `zero-shot-classification` es muy poderoso: permite que especifiques qué etiquetas usar para la clasificación, para que no dependas de las etiquetas del modelo preentrenado. Ya viste cómo el modelo puede clasificar una oración como positiva o negativa usando esas dos etiquetas — pero también puede clasificar el texto usando cualquier otro conjunto de etiquetas que definas.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

```python out
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```

Este pipeline se llama _zero-shot_ porque no necesitas ajustar el modelo con tus datos para usarlo. ¡Puede devolver directamente puntajes de probabilidad para cualquier lista de de etiquetas que escojas!

> [!TIP]
> ✏️ **¡Pruébalo!** Juega con tus propias secuencias y etiquetas, y observa cómo se comporta el modelo.


## Generación de texto

Ahora veamos cómo usar un pipeline para generar texto. La idea es que proporciones una indicación (*prompt*) y el modelo la va a completar automáticamente al generar el texto restante. Esto es parecido a la función de texto predictivo que está presente en muchos teléfonos. La generación de texto involucra aleatoriedad, por lo que es normal que no obtengas el mismo resultado que se muestra abajo.

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

```python out
[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
```

Puedes controlar cuántas secuencias diferentes se generan con el argumento `num_return_sequences` y la longitud total del texto de salida con el argumento `max_length`.

> [!TIP]
> ✏️ **¡Pruébalo!** Usa los argumentos `num_return_sequences` y `max_length` para generar dos oraciones de 15 palabras cada una.


## Usa cualquier modelo del Hub en un pipeline

Los ejemplos anteriores usaban el modelo por defecto para cada tarea, pero también puedes escoger un modelo particular del Hub y usarlo en un pipeline para una tarea específica - por ejemplo, la generación de texto. Ve al [Hub de Modelos](https://huggingface.co/models) y haz clic en la etiqueta correspondiente en la parte izquierda para mostrar únicamente los modelos soportados para esa tarea. Deberías ver una página [como esta](https://huggingface.co/models?pipeline_tag=text-generation).

¡Intentemos con el modelo [`distilgpt2`](https://huggingface.co/distilgpt2)! Puedes cargarlo en el mismo pipeline de la siguiente manera:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

```python out
[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]
```

Puedes refinar tu búsqueda de un modelo haciendo clic en las etiquetas de idioma y escoger uno que genere textos en otro idioma. El Hub de Modelos también contiene puntos de control (*checkpoints*) para modelos que soportan múltiples lenguajes.

Una vez has seleccionado un modelo haciendo clic en él, verás que hay un widget que te permite probarlo directamente en línea. De esta manera puedes probar rápidamente las capacidades del modelo antes de descargarlo.

> [!TIP]
> ✏️ **¡Pruébalo!** Usa los filtros para encontrar un modelo de generación de texto para un idioma diferente. ¡Siéntete libre de jugar con el widget y úsalo en un pipeline!


### La API de Inferencia

Todos los modelos pueden ser probados directamente en tu navegador usando la API de Inferencia, que está disponible en el [sitio web](https://huggingface.co/) de Hugging Face. Puedes jugar con el modelo directamente en esta página al pasar tu texto personalizado como entrada y ver cómo lo procesa.

La API de Inferencia que hace funcionar al widget también está disponible como un producto pago, algo útil si lo necesitas para tus flujos de trabajo. Dirígete a la [página de precios](https://huggingface.co/pricing) para más detalles.

## Llenado de ocultos (*Mask filling*)

El siguiente pipeline con el que vas a trabajar es `fill-mask`. La idea de esta tarea es llenar los espacios en blanco de un texto dado:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

```python out
[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]
```

El argumento `top_k` controla el número de posibilidades que se van a mostrar. Nota que en este caso el modelo llena la palabra especial `<mask>`, que se denomina comúnmente como *mask token*. Otros modelos pueden tener diferentes tokens, por lo que es una buena idea verificar la palabra especial adecuada cuando estés explorando diferentes modelos. Una manera de confirmar es revisar la palabra usada en el widget.

> [!TIP]
> ✏️ **¡Pruébalo!** Busca el modelo `bert-base-cased` en el Hub e identifica su *mask token* en el widget de la API de Inferencia. ¿Qué predice este modelo para la oración que está en el ejemplo de `pipeline` anterior?

## Reconocimiento de entidades nombradas

El reconocimiento de entidades nombradas (REN) es una tarea en la que el modelo tiene que encontrar cuáles partes del texto introducido corresponden a entidades como personas, ubicaciones u organizaciones. Veamos un ejemplo:

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
```

En este caso el modelo identificó correctamente que Sylvain es una persona (PER), Hugging Face una organización (ORG) y Brooklyn una ubicación (LOC).

Pasamos la opción `grouped_entities=True` en la función de creación del pipeline para decirle que agrupe las partes de la oración que corresponden a la misma entidad: Aquí el modelo agrupó correctamente "Hugging" y "Face" como una sola organización, a pesar de que su nombre está compuesto de varias palabras. De hecho, como veremos en el siguiente capítulo, el preprocesamiento puede incluso dividir palabras en partes más pequeñas. Por ejemplo, 'Sylvain' se separa en cuatro piezas: `S`, `##yl`, `##va` y`##in`. En el paso de prosprocesamiento, el pipeline reagrupa de manera exitosa dichas piezas.

> [!TIP]
> ✏️ **¡Pruébalo!** Busca en el Model Hub un modelo capaz de hacer etiquetado *part-of-speech* (que se abrevia usualmente como POS) en Inglés. ¿Qué predice este modelo para la oración en el ejemplo de arriba?

## Responder preguntas

El pipeline `question-answering` responde preguntas usando información de un contexto dado:

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

```python out
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

Observa que este pipeline funciona extrayendo información del contexto ofrecido; más no genera la respuesta.

## Resumir (*Summarization*)

Resumir es la tarea de reducir un texto en uno más corto, conservando todos (o la mayor parte de) los aspectos importantes mencionados. Aquí va un ejemplo: 

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
```

```python out
[{'summary_text': ' America has changed dramatically during recent years . The '
                  'number of engineering graduates in the U.S. has declined in '
                  'traditional engineering disciplines such as mechanical, civil '
                  ', electrical, chemical, and aeronautical engineering . Rapidly '
                  'developing economies such as China and India, as well as other '
                  'industrial countries in Europe and Asia, continue to encourage '
                  'and advance engineering .'}]
```

Similar a la generación de textos, puedes especificar los argumentos `max-length` o `min_length` para definir la longitud del resultado.


## Traducción

Para la traducción, puedes usar el modelo por defecto si indicas una pareja de idiomas en el nombre de la tarea (como `"translation_en_to_fr"`), pero la forma más sencilla es escoger el modelo que quieres usar en el [Hub de Modelos](https://huggingface.co/models). Aquí intentaremos traducir de Francés a Inglés:

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

```python out
[{'translation_text': 'This course is produced by Hugging Face.'}]
```

Al igual que los pipelines de generación de textos y resumen, puedes especificar una longitud máxima (`max_length`) o mínima (`min_length`) para el resultado.

> [!TIP]
> ✏️ **¡Pruébalo!** Busca modelos de traducción en otros idiomas e intenta traducir la oración anterior en varios de ellos.

Los pipelines vistos hasta el momento son principalmente para fines demostrativos. Fueron programados para tareas específicas y no pueden desarrollar variaciones de ellas. En el siguiente capítulo, aprenderás qué está detrás de una función `pipeline()` y cómo personalizar su comportamiento.

---

# ¿Cómo funcionan los Transformadores?


En esta sección, daremos una mirada de alto nivel a la arquitectura de los Transformadores.

## Un poco de historia sobre los Transformadores

Estos son algunos hitos en la (corta) historia de los Transformadores:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg" alt="A brief chronology of Transformers models.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono-dark.svg" alt="A brief chronology of Transformers models.">
</div>

La [arquitectura de los Transformadores](https://arxiv.org/abs/1706.03762) fue presentada por primera vez en junio de 2017. El trabajo original se enfocaba en tareas de traducción. A esto le siguió la introducción de numerosos modelos influyentes, que incluyen:

- **Junio de 2018**: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), el primer modelo de Transformadores preentrenados, que fue usado para ajustar varias tareas de PLN y obtuvo resultados de vanguardia

- **Octubre de 2018**: [BERT](https://arxiv.org/abs/1810.04805), otro gran modelo preentrenado, diseñado para producir mejores resúmenes de oraciones (¡más sobre esto en el siguiente capítulo!)

- **Febrero de 2019**: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), una versión mejorada (y más grande) de GPT, que no se liberó inmediatamente al público por consideraciones éticas

- **Octubre de 2019**: [DistilBERT](https://arxiv.org/abs/1910.01108), una versión destilada de BERT que es 60% más rápida, 40% más ligera en memoria y que retiene el 97% del desempeño de BERT

- **Octubre de 2019**: [BART](https://arxiv.org/abs/1910.13461) y [T5](https://arxiv.org/abs/1910.10683), dos grandes modelos preentrenados usando la misma arquitectura del modelo original de Transformador (los primeros en hacerlo)

- **Mayo de 2020**, [GPT-3](https://arxiv.org/abs/2005.14165), una versión aún más grande de GPT-2 con buen desempeño en una gran variedad de tareas sin la necesidad de ajustes (llamado _zero-shot learning_)

Esta lista está lejos de ser exhaustiva y solo pretende resaltar algunos de los diferentes modelos de Transformadores. De manera general, estos pueden agruparse en tres categorías:
- Parecidos a GPT (también llamados modelos _auto-regressive_)
- Parecidos a BERT (también llamados modelos _auto-encoding_)
- Parecidos a BART/T5 (también llamados modelos _sequence-to-sequence_)

Vamos a entrar en estas familias de modelos a profundidad más adelante.

## Los Transformadores son modelos de lenguaje

Todos los modelos de Transformadores mencionados con anterioridad (GPT, BERT, BART, T5, etc.) han sido entrenados como *modelos de lenguaje*. Esto significa que han sido entrenados con grandes cantidades de texto crudo de una manera auto-supervisada. El aprendizaje auto-supervisado es un tipo de entrenamiento en el que el objetivo se computa automáticamente de las entradas del modelo. ¡Esto significa que no necesitan humanos que etiqueten los datos!

Este tipo de modelos desarrolla un entendimiento estadístico del lenguaje sobre el que fue entrenado, pero no es muy útil para tareas prácticas específicas. Por lo anterior, el modelo general preentrenado pasa por un proceso llamado *transferencia de aprendizaje* (o *transfer learning* en Inglés). Durante este proceso, el modelo se ajusta de una forma supervisada -- esto es, usando etiquetas hechas por humanos -- para una tarea dada.

Un ejemplo de una tarea es predecir la palabra siguiente en una oración con base en las *n* palabras previas. Esto se denomina *modelado de lenguaje causal* porque la salida depende de las entradas pasadas y presentes, pero no en las futuras.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling-dark.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">
</div>

Otro ejemplo es el *modelado de lenguaje oculto*, en el que el modelo predice una palabra oculta en la oración.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling-dark.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">
</div>

## Los Transformadores son modelos grandes

Excepto algunos casos atípicos (como DistilBERT), la estrategia general para mejorar el desempeño es incrementar el tamaño de los modelos, así como la cantidad de datos con los que están preentrenados.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png" alt="Number of parameters of recent Transformers models" width="90%">
</div>

Desafortunadamente, entrenar un modelo, especialmente uno grande, requiere de grandes cantidades de datos. Esto se vuelve muy costoso en términos de tiempo y recursos de computación, que se traduce incluso en impacto ambiental, como se puede ver en la siguiente gráfica.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint.svg" alt="The carbon footprint of a large language model.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint-dark.svg" alt="The carbon footprint of a large language model.">
</div>


**Video:** [Ver en YouTube](https://youtu.be/ftWlj4FBHTg)


Esto es ilustrativo para un proyecto que busca un modelo (muy grande), liderado por un equipo que intenta de manera consciente reducir el impacto ambiental del preentrenamiento. La huella de ejecutar muchas pruebas para encontrar los mejores hiperparámetros es aún mayor.

Ahora imagínate si cada vez que un equipo de investigación, una organización estudiantil o una compañía intentaran entrenar un modelo, tuvieran que hacerlo desde cero. ¡Esto implicaría costos globales enormes e innecesarios!

Esta es la razón por la que compartir modelos de lenguaje es fundamental: compartir los pesos entrenados y construir sobre los existentes reduce el costo general y la huella de carbono de la comunidad.

## Transferencia de aprendizaje (*Transfer learning*)


**Video:** [Ver en YouTube](https://youtu.be/BqqfQnyjmgg)


El *preentrenamiento* es el acto de entrenar un modelo desde cero: los pesos se inicializan de manera aleatoria y el entrenamiento empieza sin un conocimiento previo.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining.svg" alt="The pretraining of a language model is costly in both time and money.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining-dark.svg" alt="The pretraining of a language model is costly in both time and money.">
</div>

Este preentrenamiento se hace usualmente sobre grandes cantidades de datos. Por lo anterior, requiere un gran corpus de datos y el entrenamiento puede tomar varias semanas.

Por su parte, el ajuste (o *fine-tuning*) es el entrenamiento realizado **después** de que el modelo ha sido preentrenado. Para hacer el ajuste, comienzas con un modelo de lenguaje preentrenado y luego realizas un aprendizaje adicional con un conjunto de datos específicos para tu tarea. Pero entonces -- ¿por qué no entrenar directamente para la tarea final? Hay un par de razones:

* El modelo preentrenado ya está entrenado con un conjunto de datos parecido al conjunto de datos de ajuste. De esta manera, el proceso de ajuste puede hacer uso del conocimiento adquirido por el modelo inicial durante el preentrenamiento (por ejemplo, para problemas de PLN, el modelo preentrenado tendrá algún tipo de entendimiento estadístico del idioma que estás usando para tu tarea).
* Dado que el modelo preentrenado fue entrenado con muchos datos, el ajuste requerirá menos datos para tener resultados decentes.
* Por la misma razón, la cantidad de tiempo y recursos necesarios para tener buenos resultados es mucho menor.

Por ejemplo, se podría aprovechar un modelo preentrenado en Inglés y después ajustarlo con un corpus arXiv, teniendo como resultado un modelo basado en investigación científica. El ajuste solo requerirá una cantidad limitada de datos: el conocimiento que el modelo preentrenado ha adquirido se "transfiere", de ahí el término *transferencia de aprendizaje*.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning-dark.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">
</div>

De ese modo, el ajuste de un modelo tendrá menos costos de tiempo, de datos, financieros y ambientales. Además es más rápido y más fácil de iterar en diferentes esquemas de ajuste, dado que el entrenamiento es menos restrictivo que un preentrenamiento completo.

Este proceso también conseguirá mejores resultados que entrenar desde cero (a menos que tengas una gran cantidad de datos), razón por la cual siempre deberías intentar aprovechar un modelo preentrenado -- uno que esté tan cerca como sea posible a la tarea respectiva -- y ajustarlo.

## Arquitectura general

En esta sección, revisaremos la arquitectura general del Transformador. No te preocupes si no entiendes algunos de los conceptos; hay secciones detalladas más adelante para cada uno de los componentes.


**Video:** [Ver en YouTube](https://youtu.be/H39Z_720T5s)


## Introducción

El modelo está compuesto por dos bloques:

* **Codificador (izquierda)**: El codificador recibe una entrada y construye una representación de ésta (sus características). Esto significa que el modelo está optimizado para conseguir un entendimiento a partir de la entrada.
* **Decodificador (derecha)**: El decodificador usa la representación del codificador (características) junto con otras entradas para generar una secuencia objetivo. Esto significa que el modelo está optimizado para generar salidas.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg" alt="Architecture of a Transformers models">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks-dark.svg" alt="Architecture of a Transformers models">
</div>

Cada una de estas partes puede ser usada de manera independiente, dependiendo de la tarea:

* **Modelos con solo codificadores**: Buenos para las tareas que requieren el entendimiento de la entrada, como la clasificación de oraciones y reconocimiento de entidades nombradas.
* **Modelos con solo decodificadores**: Buenos para tareas generativas como la generación de textos.
* **Modelos con codificadores y decodificadores** o **Modelos secuencia a secuencia**: Buenos para tareas generativas que requieren una entrada, como la traducción o resumen.

Vamos a abordar estas arquitecturas de manera independiente en secciones posteriores.

## Capas de atención

Una característica clave de los Transformadores es que están construidos con capas especiales llamadas *capas de atención*. De hecho, el título del trabajo que introdujo la arquitectura de los Transformadores fue ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Vamos a explorar los detalles de las capas de atención más adelante en el curso; por ahora, todo lo que tienes que saber es que esta capa va a indicarle al modelo que tiene que prestar especial atención a ciertas partes de la oración que le pasaste (y más o menos ignorar las demás), cuando trabaje con la representación de cada palabra.

Para poner esto en contexto, piensa en la tarea de traducir texto de Inglés a Francés. Dada la entrada "You like this course", un modelo de traducción necesitará tener en cuenta la palabra adyacente "You" para obtener la traducción correcta de la palabra "like", porque en Francés el verbo "like" se conjuga de manera distinta dependiendo del sujeto. Sin embargo, el resto de la oración no es útil para la traducción de esa palabra. En la misma línea, al traducir "this", el modelo también deberá prestar atención a la palabra "course", porque "this" se traduce de manera distinta dependiendo de si el nombre asociado es masculino o femenino. De nuevo, las otras palabras en la oración no van a importar para la traducción de "this". Con oraciones (y reglas gramaticales) más complejas, el modelo deberá prestar especial atención a palabras que pueden aparecer más lejos en la oración para traducir correctamente cada palabra.

El mismo concepto aplica para cualquier tarea asociada con lenguaje natural: una palabra por si misma tiene un significado, pero ese significado está afectado profundamente por el contexto, que puede ser cualquier palabra (o palabras) antes o después de la palabra que está siendo estudiada.

Ahora que tienes una idea de qué son las capas de atención, echemos un vistazo más de cerca a la arquitectura del Transformador.

## La arquitectura original

La arquitectura del Transformador fue diseñada originalmente para traducción. Durante el entrenamiento, el codificador recibe entradas (oraciones) en un idioma dado, mientras que el decodificador recibe las mismas oraciones en el idioma objetivo. En el codificador, las capas de atención pueden usar todas las palabras en una oración (dado que, como vimos, la traducción de una palabra dada puede ser dependiente de lo que está antes y después en la oración). Por su parte, el decodificador trabaja de manera secuencial y sólo le puede prestar atención a las palabras en la oración que ya ha traducido (es decir, sólo las palabras antes de que la palabra se ha generado). Por ejemplo, cuando hemos predicho las primeras tres palabras del objetivo de traducción se las damos al decodificador, que luego usa todas las entradas del codificador para intentar predecir la cuarta palabra.

Para acelerar el entrenamiento (cuando el modelo tiene acceso a las oraciones objetivo), al decodificador se le alimenta el objetivo completo, pero no puede usar palabras futuras (si tuviera acceso a la palabra en la posición 2 cuando trata de predecir la palabra en la posición 2, ¡el problema no sería muy difícil!). Por ejemplo, al intentar predecir la cuarta palabra, la capa de atención sólo tendría acceso a las palabras en las posiciones 1 a 3.

La arquitectura original del Transformador se veía así, con el codificador a la izquierda y el decodificador a la derecha:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg" alt="Architecture of a Transformers models">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg" alt="Architecture of a Transformers models">
</div>

Observa que la primera capa de atención en un bloque de decodificador presta atención a todas las entradas (pasadas) al decodificador, mientras que la segunda capa de atención usa la salida del codificador. De esta manera puede acceder a toda la oración de entrada para predecir de mejor manera la palabra actual. Esto es muy útil dado que diferentes idiomas pueden tener reglas gramaticales que ponen las palabras en orden distinto o algún contexto que se provee después puede ser útil para determinar la mejor traducción de una palabra dada.

La *máscara de atención* también se puede usar en el codificador/decodificador para evitar que el modelo preste atención a algunas palabras especiales --por ejemplo, la palabra especial de relleno que hace que todas las entradas sean de la misma longitud cuando se agrupan oraciones.

##  Arquitecturas vs. puntos de control

A medida que estudiemos a profundidad los Transformadores, verás menciones a *arquitecturas*, *puntos de control* (*checkpoints*) y *modelos*. Estos términos tienen significados ligeramente diferentes:

* **Arquitecturas**: Este es el esqueleto del modelo -- la definición de cada capa y cada operación que sucede al interior del modelo.
* **Puntos de control**: Estos son los pesos que serán cargados en una arquitectura dada.
* **Modelo**: Esta es un término sombrilla que no es tan preciso como "arquitectura" o "punto de control" y puede significar ambas cosas. Este curso especificará *arquitectura* o *punto de control* cuando sea relevante para evitar ambigüedades.

Por ejemplo, mientras que BERT es una arquitectura, `bert-base-cased` - un conjunto de pesos entrenados por el equipo de Google para la primera versión de BERT - es un punto de control. Sin embargo, se podría decir "el modelo BERT" y "el modelo `bert-base-cased`". 


---

# Modelos de codificadores


**Video:** [Ver en YouTube](https://youtu.be/MUqNwgPjJvQ)


Los modelos de codificadores usan únicamente el codificador del Transformador. En cada etapa, las capas de atención pueden acceder a todas las palabras de la oración inicial. Estos modelos se caracterizan generalmente por tener atención "bidireccional" y se suelen llamar modelos *auto-encoding*.

El preentrenamiento de estos modelos generalmente gira en torno a corromper de alguna manera una oración dada (por ejemplo, ocultando aleatoriamente palabras en ella) y pidiéndole al modelo que encuentre o reconstruya la oración inicial.

Los modelos de codificadores son más adecuados para tareas que requieren un entendimiento de la oración completa, como la clasificación de oraciones, reconocimiento de entidades nombradas (y más generalmente clasificación de palabras) y respuesta extractiva a preguntas.

Los miembros de esta familia de modelos incluyen:

- [ALBERT](https://huggingface.co/transformers/model_doc/albert.html)
- [BERT](https://huggingface.co/transformers/model_doc/bert.html)
- [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)
- [ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)
- [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)


---

# Modelos de decodificadores


**Video:** [Ver en YouTube](https://youtu.be/d_ixlCubqQw)


Los modelos de decodificadores usan únicamente el decodificador del Transformador. En cada etapa, para una palabra dada las capas de atención pueden acceder solamente a las palabras que se ubican antes en la oración. Estos modelos se suelen llamar modelos *auto-regressive*.

El preentrenamiento de los modelos de decodificadores generalmente gira en torno a la predicción de la siguiente palabra en la oración.

Estos modelos son más adecuados para tareas que implican la generación de texto.

Los miembros de esta familia de modelos incluyen:

- [CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)
- [GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)
- [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
- [Transformer XL](https://huggingface.co/transformers/model_doc/transformerxl.html)


---

# Modelos secuencia a secuencia


**Video:** [Ver en YouTube](https://youtu.be/0_4KEb08xrE)


Los modelos codificador/decodificador (también llamados *modelos secuencia a secuencia*) usan ambas partes de la arquitectura del Transformador. En cada etapa, las capas de atención del codificador pueden acceder a todas las palabras de la secuencia inicial, mientras que las capas de atención del decodificador sólo pueden acceder a las palabras que se ubican antes de una palabra dada en el texto de entrada.

El preentrenamiento de estos modelos se puede hacer usando los objetivos de los modelos de codificadores o decodificadores, pero usualmente implican algo más complejo. Por ejemplo, [T5](https://huggingface.co/t5-base) está preentrenado al reemplazar segmentos aleatorios de texto (que pueden contener varias palabras) con una palabra especial que las oculta, y el objetivo es predecir el texto que esta palabra reemplaza.

Los modelos secuencia a secuencia son más adecuados para tareas relacionadas con la generación de nuevas oraciones dependiendo de una entrada dada, como resumir, traducir o responder generativamente preguntas.

Algunos miembros de esta familia de modelos son:

- [BART](https://huggingface.co/transformers/model_doc/bart.html)
- [mBART](https://huggingface.co/transformers/model_doc/mbart.html)
- [Marian](https://huggingface.co/transformers/model_doc/marian.html)
- [T5](https://huggingface.co/transformers/model_doc/t5.html)


---

# Sesgos y limitaciones


Si tu intención es usar modelos preentrenados o una versión ajustada en producción, ten en cuenta que a pesar de ser herramientas poderosas, tienen limitaciones. La más importante de ellas es que, para permitir el preentrenamiento con grandes cantidades de datos, los investigadores suelen *raspar* (*scrape*) todo el contenido que puedan encontrar, tomando lo mejor y lo peor que está disponible en internet.

Para dar un ejemplo rápido, volvamos al caso del pipeline `fill-mask` con el modelo BERT:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
```

```python out
['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

Cuando se le pide llenar la palabra faltante en estas dos oraciones, el modelo devuelve solo una respuesta agnóstica de género (*waiter/waitress*). Las otras son ocupaciones que se suelen asociar con un género específico -- y si, prostituta es una de las primeras 5 posibilidades que el modelo asocia con "mujer" y "trabajo". Esto sucede a pesar de que BERT es uno de los pocos modelos de Transformadores que no se construyeron basados en datos *raspados* de todo el internet, pero usando datos aparentemente neutrales (está entrenado con los conjuntos de datos de [Wikipedia en Inglés](https://huggingface.co/datasets/wikipedia) y [BookCorpus](https://huggingface.co/datasets/bookcorpus)).

Cuando uses estas herramientas, debes tener en cuenta que el modelo original que estás usando puede muy fácilmente generar contenido sexista, racista u homófobo. Ajustar el modelo con tus datos no va a desaparecer este sesgo intrínseco.


---

# Resumen


En este capítulo viste cómo abordar diferentes tareas de PLN usando la función de alto nivel `pipeline()` de 🤗 Transformers. También viste como buscar modelos en el Hub, así como usar la API de Inferencia para probar los modelos directamente en tu navegador.

Discutimos brevemente el funcionamiento de los Transformadores y hablamos sobre la importancia de la transferencia de aprendizaje y el ajuste. Un aspecto clave es que puedes usar la arquitectura completa o sólo el codificador o decodificador, dependiendo de qué tipo de tarea quieres resolver. La siguiente tabla resume lo anterior:

| Modelo                    | Ejemplos                                   | Tareas                                                                                              |
|---------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Codificador               | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Clasificación de oraciones, reconocimiento de entidades nombradas, respuesta extractiva a preguntas |
| Decodificador             | CTRL, GPT, GPT-2, Transformer XL           | Generación de texto                                                                                 |
| Codificador-decodificador | BART, T5, Marian, mBART                    | Resumen, traducción, respuesta generativa a preguntas                                               |


---



# Quiz de final de capítulo


¡Este capítulo cubrió una gran variedad de temas! No te preocupes si no entendiste todos los detalles; los siguientes capítulos te ayudarán a entender cómo funcionan las cosas detrás de cámaras.

Por ahora, ¡revisemos lo que aprendiste en este capítulo!

### 1. Explora el Hub y busca el punto de control `roberta-large-mnli`. ¿Qué tarea desarrolla?


- Resumen
- Clasificación de texto
- Generación de texto


### 2. ¿Qué devuelve el siguiente código?

```py
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```


- Devuelve los puntajes de clasificación de esta oración, con las etiquetas \
- Devuelve un texto generado que completa esta oración.
- Devuelve las palabras que representan personas, organizaciones o ubicaciones.


### 3. ¿Qué debería reemplazar ... en este ejemplo de código?

```py
from transformers import pipeline

filler = pipeline("fill-mask", model="bert-base-cased")
result = filler("...")
```


- This &#60;mask> has been waiting for you.
- This [MASK] has been waiting for you.
- This man has been waiting for you.


### 4. ¿Por qué fallará este código?

```py
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")
```


- Este pipeline necesita que se le indiquen etiquetas para clasificar el texto.
- Este pipeline requiere varias oraciones, no sólo una.
- La librería 🤗 Transformers está dañada, como siempre.
- Este pipeline necesita entradas más largas; esta oración es muy corta.


### 5. ¿Qué significa "transferencia de aprendizaje"?


- Transferir el conocimiento de un modelo preentrenado a un nuevo modelo, al entrenarlo en el mismo conjunto de datos.
- Transferir el conocimiento de un modelo preentrenado a un nuevo modelo al inicializar un segundo modelo con los pesos del primero.
- Transferir el conocimiento de un modelo preentrenado al construir el segundo modelo con la misma arquitectura del primero.


### 6. ¿Verdadero o falso? Un modelo de lenguaje usualmente no necesita etiquetas para su preentrenamiento.


- Verdadero
- Falso


### 7. Selecciona la oración que describe mejor los términos "modelo", "arquitectura" y "pesos".


- Si un modelo es un edificio, su arquitectura es el plano y los pesos son las personas que viven allí.
- Una arquitectura es un mapa para construir un modelo y sus pesos son las ciudades representadas en el mapa.
- Una arquitectura es una sucesión de funciones matemáticas para construir un modelo y sus pesos son los parámetros de dichas funciones.


### 8. ¿Cuál de los siguientes tipos de modelos usarías para completar una indicación con texto generado?


- Un modelo de codificadores
- Un modelo de decodificadores
- Un modelo secuencia a secuencia


### 9. ¿Cuál de los siguientes tipos de modelos usarías para resumir textos?


- Un modelo de codificadores
- Un modelo de decodificadores
- Un modelo secuencia a secuencia


### 10. ¿Cuál de los siguientes tipos de modelos usarías para clasificar texto de acuerdo con ciertas etiquetas?


- Un modelo de codificadores
- Un modelo de decodificadores
- Un modelo secuencia a secuencia


### 11. ¿Cuál puede ser una posible fuente del sesgo observado en un modelo?


- El modelo es una versión ajustada de un modelo preentrenado y tomó el sesgo a partir de allí.
- Los datos con los que se entrenó el modelo están sesgados.
- La métrica que el modelo estaba optimizando está sesgada.



---

# Es hora del examen!

Es hora de poner a prueba tus conocimientos! Hemos preparado un breve cuestionario para evaluar tu comprension de los conceptos cubiertos en este capitulo.

Para realizar el cuestionario, deberas seguir estos pasos:

1. Inicia sesion en tu cuenta de Hugging Face.
2. Responde las preguntas del cuestionario.
3. Envia tus respuestas.


## Cuestionario de opcion multiple

En este cuestionario, se te pedira que selecciones la respuesta correcta de una lista de opciones. Te evaluaremos sobre los fundamentos del ajuste fino supervisado.

<iframe
	src="https://huggingface-course-chapter-1-exam.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>


---

# 2. Usando 🤗 Transformers

# Introduccion[[introduction]]


Como viste en el [Capitulo 1](/course/chapter1), los modelos Transformer suelen ser muy grandes. Con millones a decenas de *miles de millones* de parametros, entrenar y desplegar estos modelos es una tarea complicada. Ademas, con nuevos modelos siendo lanzados casi a diario y cada uno con su propia implementacion, probarlos todos no es tarea facil.

La biblioteca 🤗 Transformers fue creada para resolver este problema. Su objetivo es proporcionar una API unica a traves de la cual cualquier modelo Transformer pueda ser cargado, entrenado y guardado. Las principales caracteristicas de la biblioteca son:

- **Facilidad de uso**: Descargar, cargar y usar un modelo de NLP de ultima generacion para inferencia se puede hacer en solo dos lineas de codigo.
- **Flexibilidad**: En su nucleo, todos los modelos son clases simples de PyTorch `nn.Module` y pueden ser manejados como cualquier otro modelo en sus respectivos frameworks de machine learning (ML).
- **Simplicidad**: Casi no se hacen abstracciones en toda la biblioteca. El concepto "Todo en un archivo" es fundamental: el forward pass de un modelo esta definido completamente en un solo archivo, de modo que el codigo mismo sea comprensible y modificable.

Esta ultima caracteristica hace que 🤗 Transformers sea bastante diferente de otras bibliotecas de ML. Los modelos no estan construidos sobre modulos que se comparten entre archivos; en su lugar, cada modelo tiene sus propias capas. Ademas de hacer los modelos mas accesibles y comprensibles, esto te permite experimentar facilmente con un modelo sin afectar a otros.

Este capitulo comenzara con un ejemplo de principio a fin donde usamos un modelo y un tokenizador juntos para replicar la funcion `pipeline()` introducida en el [Capitulo 1](/course/chapter1). A continuacion, discutiremos la API del modelo: profundizaremos en las clases de modelo y configuracion, y te mostraremos como cargar un modelo y como procesa entradas numericas para producir predicciones.

Luego veremos la API del tokenizador, que es el otro componente principal de la funcion `pipeline()`. Los tokenizadores se encargan del primer y ultimo paso del procesamiento, manejando la conversion de texto a entradas numericas para la red neuronal, y la conversion de vuelta a texto cuando es necesario. Finalmente, te mostraremos como manejar el envio de multiples oraciones a traves de un modelo en un lote preparado, y luego cerraremos con una mirada mas cercana a la funcion de alto nivel `tokenizer()`.

> [!TIP]
> ⚠️ Para beneficiarte de todas las funciones disponibles con el Model Hub y 🤗 Transformers, te recomendamos <a href="https://huggingface.co/join">crear una cuenta</a>.


---



# Detras del pipeline[[behind-the-pipeline]]


**Video:** [Ver en YouTube](https://youtu.be/1pedAIvTWXk)


Comencemos con un ejemplo completo, observando lo que sucedio detras de escena cuando ejecutamos el siguiente codigo en el [Capitulo 1](/course/chapter1):

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

y obtuvimos:

```python out
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

Como vimos en el [Capitulo 1](/course/chapter1), este pipeline agrupa tres pasos: preprocesamiento, pasar las entradas por el modelo y postprocesamiento:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline.svg" alt="El pipeline completo de NLP: tokenizacion del texto, conversion a IDs, e inferencia a traves del modelo Transformer y la cabeza del modelo."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline-dark.svg" alt="El pipeline completo de NLP: tokenizacion del texto, conversion a IDs, e inferencia a traves del modelo Transformer y la cabeza del modelo."/>
</div>

Repasemos rapidamente cada uno de estos.

## Preprocesamiento con un tokenizador[[preprocessing-with-a-tokenizer]]

Como otras redes neuronales, los modelos Transformer no pueden procesar texto crudo directamente, por lo que el primer paso de nuestro pipeline es convertir las entradas de texto en numeros que el modelo pueda entender. Para hacer esto usamos un *tokenizador*, que sera responsable de:

- Dividir la entrada en palabras, subpalabras o simbolos (como signos de puntuacion) que se llaman *tokens*
- Mapear cada token a un entero
- Agregar entradas adicionales que puedan ser utiles para el modelo

Todo este preprocesamiento debe hacerse exactamente de la misma manera que cuando el modelo fue preentrenado, por lo que primero necesitamos descargar esa informacion desde el [Model Hub](https://huggingface.co/models). Para hacer esto, usamos la clase `AutoTokenizer` y su metodo `from_pretrained()`. Usando el nombre del checkpoint de nuestro modelo, automaticamente obtendra los datos asociados con el tokenizador del modelo y los almacenara en cache (para que solo se descarguen la primera vez que ejecutes el codigo a continuacion).

Como el checkpoint predeterminado del pipeline `sentiment-analysis` es `distilbert-base-uncased-finetuned-sst-2-english` (puedes ver su tarjeta de modelo [aqui](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)), ejecutamos lo siguiente:

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Una vez que tenemos el tokenizador, podemos pasarle directamente nuestras oraciones y obtendremos un diccionario listo para alimentar a nuestro modelo! Lo unico que queda por hacer es convertir la lista de IDs de entrada a tensores.

Puedes usar 🤗 Transformers sin preocuparte por que framework de ML se usa como backend; podria ser PyTorch o Flax para algunos modelos. Sin embargo, los modelos Transformer solo aceptan *tensores* como entrada. Si es la primera vez que escuchas sobre tensores, puedes pensar en ellos como arreglos de NumPy. Un arreglo de NumPy puede ser un escalar (0D), un vector (1D), una matriz (2D), o tener mas dimensiones. Efectivamente es un tensor; los tensores de otros frameworks de ML se comportan de manera similar, y generalmente son tan simples de instanciar como los arreglos de NumPy.

Para especificar el tipo de tensores que queremos obtener (PyTorch o NumPy simple), usamos el argumento `return_tensors`:

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

No te preocupes por el padding y truncation todavia; los explicaremos mas adelante. Lo principal a recordar aqui es que puedes pasar una oracion o una lista de oraciones, asi como especificar el tipo de tensores que quieres obtener (si no se pasa ningun tipo, obtendras una lista de listas como resultado).

Asi es como se ven los resultados como tensores de PyTorch:

```python out
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

La salida en si es un diccionario que contiene dos claves, `input_ids` y `attention_mask`. `input_ids` contiene dos filas de enteros (una para cada oracion) que son los identificadores unicos de los tokens en cada oracion. Explicaremos que es `attention_mask` mas adelante en este capitulo.

## Pasando por el modelo[[going-through-the-model]]

Podemos descargar nuestro modelo preentrenado de la misma manera que lo hicimos con nuestro tokenizador. 🤗 Transformers proporciona una clase `AutoModel` que tambien tiene un metodo `from_pretrained()`:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

En este fragmento de codigo, hemos descargado el mismo checkpoint que usamos en nuestro pipeline antes (en realidad ya deberia estar en cache) e instanciamos un modelo con el.

Esta arquitectura contiene solo el modulo Transformer base: dadas algunas entradas, produce lo que llamaremos *estados ocultos*, tambien conocidos como *caracteristicas*. Para cada entrada del modelo, recuperaremos un vector de alta dimension que representa la **comprension contextual de esa entrada por el modelo Transformer**.

Si esto no tiene sentido, no te preocupes. Lo explicaremos todo mas adelante.

Aunque estos estados ocultos pueden ser utiles por si mismos, usualmente son entradas a otra parte del modelo, conocida como la *cabeza*. En el [Capitulo 1](/course/chapter1), las diferentes tareas podrian haber sido realizadas con la misma arquitectura, pero cada una de estas tareas tendra una cabeza diferente asociada.

### Un vector de alta dimension?[[a-high-dimensional-vector]]

El vector producido por el modulo Transformer es usualmente grande. Generalmente tiene tres dimensiones:

- **Tamano del lote**: El numero de secuencias procesadas a la vez (2 en nuestro ejemplo).
- **Longitud de secuencia**: La longitud de la representacion numerica de la secuencia (16 en nuestro ejemplo).
- **Tamano oculto**: La dimension del vector de cada entrada del modelo.

Se dice que es de "alta dimension" por el ultimo valor. El tamano oculto puede ser muy grande (768 es comun para modelos mas pequenos, y en modelos mas grandes esto puede alcanzar 3072 o mas).

Podemos ver esto si alimentamos las entradas que preprocesamos a nuestro modelo:

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```python out
torch.Size([2, 16, 768])
```

Nota que las salidas de los modelos 🤗 Transformers se comportan como `namedtuple`s o diccionarios. Puedes acceder a los elementos por atributos (como hicimos) o por clave (`outputs["last_hidden_state"]`), o incluso por indice si sabes exactamente donde esta lo que buscas (`outputs[0]`).

### Cabezas de modelo: Dando sentido a los numeros[[model-heads-making-sense-out-of-numbers]]

Las cabezas del modelo toman el vector de alta dimension de estados ocultos como entrada y los proyectan a una dimension diferente. Usualmente estan compuestas de una o pocas capas lineales:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg" alt="Una red Transformer junto con su cabeza."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head-dark.svg" alt="Una red Transformer junto con su cabeza."/>
</div>

La salida del modelo Transformer se envia directamente a la cabeza del modelo para ser procesada.

En este diagrama, el modelo esta representado por su capa de embeddings y las capas subsiguientes. La capa de embeddings convierte cada ID de entrada en la entrada tokenizada a un vector que representa el token asociado. Las capas subsiguientes manipulan esos vectores usando el mecanismo de atencion para producir la representacion final de las oraciones.

Hay muchas arquitecturas diferentes disponibles en 🤗 Transformers, con cada una disenada para abordar una tarea especifica. Aqui hay una lista no exhaustiva:

- `*Model` (recupera los estados ocultos)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- y otros 🤗

Para nuestro ejemplo, necesitaremos un modelo con una cabeza de clasificacion de secuencias (para poder clasificar las oraciones como positivas o negativas). Entonces, no usaremos realmente la clase `AutoModel`, sino `AutoModelForSequenceClassification`:

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

Ahora si miramos la forma de nuestras salidas, la dimensionalidad sera mucho menor: la cabeza del modelo toma como entrada los vectores de alta dimension que vimos antes, y produce vectores que contienen dos valores (uno por etiqueta):

```python
print(outputs.logits.shape)
```

```python out
torch.Size([2, 2])
```

Como solo tenemos dos oraciones y dos etiquetas, el resultado que obtenemos de nuestro modelo tiene forma 2 x 2.

## Postprocesamiento de la salida[[postprocessing-the-output]]

Los valores que obtenemos como salida de nuestro modelo no necesariamente tienen sentido por si mismos. Echemos un vistazo:

```python
print(outputs.logits)
```

```python out
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

Nuestro modelo predijo `[-1.5607, 1.6123]` para la primera oracion y `[ 4.1692, -3.3464]` para la segunda. Esos no son probabilidades sino *logits*, las puntuaciones crudas, no normalizadas, producidas por la ultima capa del modelo. Para ser convertidos a probabilidades, necesitan pasar por una capa [SoftMax](https://es.wikipedia.org/wiki/Funci%C3%B3n_SoftMax) (todos los modelos 🤗 Transformers producen logits, ya que la funcion de perdida para el entrenamiento generalmente fusionara la ultima funcion de activacion, como SoftMax, con la funcion de perdida real, como la entropia cruzada):

```py
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```python out
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

Ahora podemos ver que el modelo predijo `[0.0402, 0.9598]` para la primera oracion y `[0.9995, 0.0005]` para la segunda. Estas son puntuaciones de probabilidad reconocibles.

Para obtener las etiquetas correspondientes a cada posicion, podemos inspeccionar el atributo `id2label` de la configuracion del modelo (mas sobre esto en la siguiente seccion):

```python
model.config.id2label
```

```python out
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

Ahora podemos concluir que el modelo predijo lo siguiente:

- Primera oracion: NEGATIVE: 0.0402, POSITIVE: 0.9598
- Segunda oracion: NEGATIVE: 0.9995, POSITIVE: 0.0005

Hemos reproducido exitosamente los tres pasos del pipeline: preprocesamiento con tokenizadores, pasar las entradas por el modelo y postprocesamiento! Ahora tomemos algo de tiempo para profundizar en cada uno de esos pasos.

> [!TIP]
> ✏️ **Pruebalo!** Elige dos (o mas) textos propios y pasalos por el pipeline de `sentiment-analysis`. Luego replica los pasos que viste aqui tu mismo y verifica que obtienes los mismos resultados!


---



# Modelos[[the-models]]


**Video:** [Ver en YouTube](https://youtu.be/AhChOFRegn4)


En esta seccion, examinaremos mas de cerca la creacion y el uso de modelos. Usaremos la clase `AutoModel`, que es practica cuando quieres instanciar cualquier modelo desde un checkpoint.

## Creando un Transformer[[creating-a-transformer]]

Comencemos examinando lo que sucede cuando instanciamos un `AutoModel`:

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

Similar al tokenizador, el metodo `from_pretrained()` descargara y almacenara en cache los datos del modelo desde Hugging Face Hub. Como se menciono anteriormente, el nombre del checkpoint corresponde a una arquitectura y pesos de modelo especificos, en este caso un modelo BERT con una arquitectura basica (12 capas, 768 de tamano oculto, 12 cabezas de atencion) y entradas sensibles a mayusculas/minusculas (lo que significa que la distincion entre mayusculas y minusculas es importante). Hay muchos checkpoints disponibles en el Hub — puedes explorarlos [aqui](https://huggingface.co/models).

La clase `AutoModel` y sus asociadas son en realidad simples envoltorios disenados para obtener la arquitectura de modelo apropiada para un checkpoint dado. Es una clase "auto" que significa que adivinara la arquitectura de modelo apropiada para ti e instanciara la clase de modelo correcta. Sin embargo, si conoces el tipo de modelo que quieres usar, puedes usar la clase que define su arquitectura directamente:

```py
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

## Cargando y guardando[[loading-and-saving]]

Guardar un modelo es tan simple como guardar un tokenizador. De hecho, los modelos tienen el mismo metodo `save_pretrained()`, que guarda los pesos y la configuracion de la arquitectura del modelo:

```py
model.save_pretrained("directory_on_my_computer")
```

Esto guardara dos archivos en tu disco:

```
ls directory_on_my_computer

config.json model.safetensors
```

Si miras dentro del archivo *config.json*, veras todos los atributos necesarios para construir la arquitectura del modelo. Este archivo tambien contiene algunos metadatos, como de donde se origino el checkpoint y que version de 🤗 Transformers estabas usando cuando guardaste el checkpoint por ultima vez.

El archivo *pytorch_model.safetensors* se conoce como el diccionario de estado; contiene todos los pesos de tu modelo. Los dos archivos trabajan juntos: el archivo de configuracion es necesario para conocer la arquitectura del modelo, mientras que los pesos del modelo son los parametros del modelo.

Para reutilizar un modelo guardado, usa el metodo `from_pretrained()` nuevamente:

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")
```

Una caracteristica maravillosa de la biblioteca 🤗 Transformers es la capacidad de compartir facilmente modelos y tokenizadores con la comunidad. Para hacer esto, asegurate de tener una cuenta en [Hugging Face](https://huggingface.co). Si estas usando un notebook, puedes iniciar sesion facilmente con esto:

```python
from huggingface_hub import notebook_login

notebook_login()
```

De lo contrario, en tu terminal ejecuta:

```bash
huggingface-cli login
```

Luego puedes subir el modelo al Hub con el metodo `push_to_hub()`:

```py
model.push_to_hub("my-awesome-model")
```

Esto subira los archivos del modelo al Hub, en un repositorio bajo tu espacio de nombres llamado *my-awesome-model*. Entonces, cualquiera puede cargar tu modelo con el metodo `from_pretrained()`!

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/my-awesome-model")
```

Puedes hacer mucho mas con la API del Hub:
- Subir un modelo desde un repositorio local
- Actualizar archivos especificos sin resubir todo
- Agregar tarjetas de modelo para documentar las capacidades, limitaciones, sesgos conocidos del modelo, etc.

Consulta [la documentacion](https://huggingface.co/docs/huggingface_hub/how-to-upstream) para un tutorial completo sobre esto, o revisa el [Capitulo 4](/course/chapter4) avanzado.

## Codificando texto[[encoding-text]]

Los modelos Transformer manejan texto convirtiendolo a numeros. Aqui veremos exactamente lo que sucede cuando tu texto es procesado por el tokenizador. Ya hemos visto en el [Capitulo 1](/course/chapter1) que los tokenizadores dividen el texto en tokens y luego convierten estos tokens en numeros. Podemos ver esta conversion a traves de un tokenizador simple:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

```python out
{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Obtenemos un diccionario con los siguientes campos:
- input_ids: representaciones numericas de tus tokens
- token_type_ids: estos le dicen al modelo que parte de la entrada es la oracion A y cual es la oracion B (se discute mas en la siguiente seccion)
- attention_mask: esto indica que tokens deben ser atendidos y cuales no (se discute mas adelante)

Podemos decodificar los IDs de entrada para obtener el texto original:

```py
tokenizer.decode(encoded_input["input_ids"])
```

```python out
"[CLS] Hello, I'm a single sentence! [SEP]"
```

Notaras que el tokenizador ha agregado tokens especiales — `[CLS]` y `[SEP]` — requeridos por el modelo. No todos los modelos necesitan tokens especiales; se utilizan cuando un modelo fue preentrenado con ellos, en cuyo caso el tokenizador necesita agregarlos ya que el modelo los espera.

Puedes codificar multiples oraciones a la vez, ya sea agrupandolas juntas (lo discutiremos pronto) o pasando una lista:

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)
```

```python out
{'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

Nota que al pasar multiples oraciones, el tokenizador devuelve una lista para cada oracion por cada valor del diccionario. Tambien podemos pedirle al tokenizador que devuelva tensores directamente de PyTorch:

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)
```

```python out
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

Pero hay un problema: las dos listas no tienen la misma longitud! Los arreglos y tensores necesitan ser rectangulares, por lo que no podemos simplemente convertir estas listas a un tensor de PyTorch (o arreglo de NumPy). El tokenizador proporciona una opcion para eso: padding.

### Rellenando entradas[[padding-inputs]]

Si le pedimos al tokenizador que rellene las entradas, hara que todas las oraciones tengan la misma longitud agregando un token de relleno especial a las oraciones que son mas cortas que la mas larga:

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)
```

```python out
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

Ahora tenemos tensores rectangulares! Nota que los tokens de relleno han sido codificados en IDs de entrada con ID 0, y tienen un valor de mascara de atencion de 0 tambien. Esto es porque esos tokens de relleno no deben ser analizados por el modelo: no son parte de la oracion real.

### Truncando entradas[[truncating-inputs]]

Los tensores pueden volverse demasiado grandes para ser procesados por el modelo. Por ejemplo, BERT solo fue preentrenado con secuencias de hasta 512 tokens, por lo que no puede procesar secuencias mas largas. Si tienes secuencias mas largas de lo que el modelo puede manejar, necesitaras truncarlas con el parametro `truncation`:

```py
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])
```

```python out
[101, 1188, 1110, 170, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1179, 5650, 119, 102]
```

Combinando los argumentos de padding y truncation, puedes asegurarte de que tus tensores tengan el tamano exacto que necesitas:

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
```

```python out
{'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
         [  101,  1045,  1005,  1049,   102]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])}
```

### Agregando tokens especiales

Los tokens especiales (o al menos el concepto de ellos) son particularmente importantes para BERT y modelos derivados. Estos tokens se agregan para representar mejor los limites de las oraciones, como el comienzo de una oracion (`[CLS]`) o el separador entre oraciones (`[SEP]`). Veamos un ejemplo simple:

```py
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])
```

```python out
[101, 1731, 1132, 1128, 136, 102]
'[CLS] How are you? [SEP]'
```

Estos tokens especiales son agregados automaticamente por el tokenizador. No todos los modelos necesitan tokens especiales; se usan principalmente cuando un modelo fue preentrenado con ellos, en cuyo caso el tokenizador los agregara ya que el modelo los espera.

### Por que es necesario todo esto?

Aqui hay un ejemplo concreto. Considera estas secuencias codificadas:

```py
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
```

Una vez tokenizadas, tenemos:

```python
encoded_sequences = [
    [
        101,
        1045,
        1005,
        2310,
        2042,
        3403,
        2005,
        1037,
        17662,
        12172,
        2607,
        2026,
        2878,
        2166,
        1012,
        102,
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]
```

Esta es una lista de secuencias codificadas: una lista de listas. Los tensores solo aceptan formas rectangulares (piensa en matrices). Este "arreglo" ya tiene forma rectangular, por lo que convertirlo a un tensor es facil:

```py
import torch

model_inputs = torch.tensor(encoded_sequences)
```

### Usando los tensores como entradas al modelo[[using-the-tensors-as-inputs-to-the-model]]

Usar los tensores con el modelo es extremadamente simple — solo llamamos al modelo con las entradas:

```py
output = model(model_inputs)
```

Aunque el modelo acepta muchos argumentos diferentes, solo los IDs de entrada son necesarios. Explicaremos que hacen los otros argumentos y cuando son requeridos mas adelante, pero primero necesitamos examinar mas de cerca los tokenizadores que construyen las entradas que un modelo Transformer puede entender.


---



# Tokenizadores


**Video:** [Ver en YouTube](https://youtu.be/VFp38yj8h3A)


Los tokenizadores son uno de los componentes fundamentales del pipeline en NLP. Sirven para traducir texto en datos que los modelos puedan procesar; es decir, de texto a valores numéricos. En esta sección veremos en qué se fundamenta todo el proceso de tokenizado.

En las tareas de NLP, los datos generalmente ingresan como texto crudo. Por ejemplo:

```
Jim Henson era un titiritero
```

Sin embargo, necesitamos una forma de convertir el texto crudo a valores numéricos para los modelos. Eso es precisamente lo que hacen los tokenizadores, y existe una variedad de formas en que puede hacerse. El objetivo final es obtener valores que sean cortos pero muy significativos para el modelo. 

Veamos algunos algoritmos de tokenización, e intentemos atacar algunas preguntas que puedas tener.

## Tokenización Word-based


**Video:** [Ver en YouTube](https://youtu.be/nhJxYji1aho)


El primer tokenizador que nos ocurre es el _word-based_ (_basado-en-palabras_). Es generalmente sencillo, con pocas normas, y generalmente da buenos resultados. Por ejemplo, en la imagen a continuación separamos el texto en palabras y buscamos una representación numérica.

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/word_based_tokenization.svg" alt="Un ejemplo de tokenizador _word-based_."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/word_based_tokenization-dark.svg" alt="Un ejemplo de tokenizador _word-based_."/>
</div>

Existen varias formas de separar el texto. Por ejemplo, podríamos usar los espacios para tokenizar usando Python y la función `split()`. 

```py
tokenized_text = "Jim Henson era un titiritero".split()
print(tokenized_text)
```

```python out
['Jim', 'Henson', 'era', 'un', 'titiritero']
```

También hay variaciones de tokenizadores de palabras que tienen reglas adicionales para la puntuación. Con este tipo de tokenizador, podemos acabar con unos "vocabularios" bastante grandes, donde un vocabulario se define por el número total de tokens independientes que tenemos en nuestro corpus.

A cada palabra se le asigna un ID, empezando por 0 y subiendo hasta el tamaño del vocabulario. El modelo utiliza estos ID para identificar cada palabra.

Si queremos cubrir completamente un idioma con un tokenizador basado en palabras, necesitaremos tener un identificador para cada palabra del idioma, lo que generará una enorme cantidad de tokens. Por ejemplo, hay más de 500.000 palabras en el idioma inglés, por lo que para construir un mapa de cada palabra a un identificador de entrada necesitaríamos hacer un seguimiento de esa cantidad de identificadores. Además, palabras como "perro" se representan de forma diferente a palabras como "perros", y el modelo no tendrá forma de saber que "perro" y "perros" son similares: identificará las dos palabras como no relacionadas. Lo mismo ocurre con otras palabras similares, como "correr" y "corriendo", que el modelo no verá inicialmente como similares.

Por último, necesitamos un token personalizado para representar palabras que no están en nuestro vocabulario. Esto se conoce como el token "desconocido", a menudo representado como "[UNK]" o "&lt;unk&gt;". Generalmente, si el tokenizador está produciendo muchos de estos tokens es una mala señal, ya que no fue capaz de recuperar una representación de alguna palabra y está perdiendo información en el proceso. El objetivo al elaborar el vocabulario es hacerlo de tal manera que el tokenizador tokenice el menor número de palabras posibles en tokens desconocidos.

Una forma de reducir la cantidad de tokens desconocidos es ir un poco más allá, utilizando un tokenizador _word-based_.

## Tokenización Character-based


**Video:** [Ver en YouTube](https://youtu.be/ssLq_EK2jLE)


Un tokenizador _character-based_ separa el texto en caracteres, y no en palabras. Esto conlleva dos beneficios principales:

- Obtenemos un vocabulario mucho más corto.
- Habrá muchos menos tokens por fuera del vocabulario conocido.

No obstante, pueden surgir inconvenientes por los espacios en blanco y signos de puntuación.

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/character_based_tokenization.svg" alt="Ejemplo de tokenizador basado en palabras."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/character_based_tokenization-dark.svg" alt="Ejemplo de tokenizador basado en palabras."/>
</div>

Así, este método tampoco es perfecto. Dada que la representación se construyó con caracteres, uno podría pensar intuitivamente que resulta menos significativo: Cada una de las palabras no significa mucho por separado, mientras que las palabras sí. Sin embargo, eso es dependiente del idioma. Por ejemplo en Chino, cada uno de los caracteres conlleva más información que en un idioma latino.

Otro aspecto a considerar es que terminamos con una gran cantidad de tokens que el modelo debe procesar, mientras que en el caso del tokenizador _word-based_, un token representa una palabra, en la representación de caracteres fácilmente puede necesitar más de 10 tokens.

Para obtener lo mejor de ambos mundos, podemos usar una combinación de las técnicas: la tokenización por *subword tokenization*.

## Tokenización por Subword


**Video:** [Ver en YouTube](https://youtu.be/zHvTiHr506c)


Los algoritmos de tokenización de subpalabras se basan en el principio de que las palabras de uso frecuente no deben dividirse, mientras que las palabras raras deben descomponerse en subpalabras significativas.

Por ejemplo, "extrañamente" podría considerarse una palabra rara y podría descomponerse en "extraña" y "mente". Es probable que ambas aparezcan con más frecuencia como subpalabras independientes, mientras que al mismo tiempo el significado de "extrañamente" se mantiene por el significado compuesto de "extraña" y "mente".

Este es un ejemplo que muestra cómo un algoritmo de tokenización de subpalabras tokenizaría la secuencia "Let's do tokenization!":

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword.svg" alt="Un tokenizador basado en subpalabras."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword-dark.svg" alt="Un tokenizador basado en subpalabras."/>
</div>

Estas subpalabras terminan aportando mucho significado semántico: por ejemplo, en el ejemplo anterior, "tokenización" se dividió en "token" y "ización", dos tokens que tienen un significado semántico y a la vez son eficientes en cuanto al espacio (sólo se necesitan dos tokens para representar una palabra larga). Esto nos permite tener una cobertura relativamente buena con vocabularios pequeños y casi sin tokens desconocidos.

Este enfoque es especialmente útil en algunos idiomas como el turco, donde se pueden formar palabras complejas (casi) arbitrariamente largas encadenando subpalabras.

### Y más!

Como es lógico, existen muchas más técnicas. Por nombrar algunas:

- Byte-level BPE (a nivel de bytes), como usa GPT-2
- WordPiece, usado por BERT
- SentencePiece or Unigram (pedazo de sentencia o unigrama), como se usa en los modelos multilingües

A este punto, deberías tener conocimientos suficientes sobre el funcionamiento de los tokenizadores para empezar a utilizar la API.

## Cargando y guardando

Cargar y guardar tokenizadores es tan sencillo como lo es con los modelos. En realidad, se basa en los mismos dos métodos: `from_pretrained()` y `save_pretrained()`. Estos métodos cargarán o guardarán el algoritmo utilizado por el tokenizador (un poco como la *arquitectura* del modelo) así como su vocabulario (un poco como los *pesos* del modelo).

La carga del tokenizador BERT entrenado con el mismo punto de control que BERT se realiza de la misma manera que la carga del modelo, excepto que utilizamos la clase `BertTokenizer`:

```py
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```


**PyTorch:**

Al igual que `AutoModel`, la clase `AutoTokenizer` tomará la clase de tokenizador adecuada en la librería basada en el nombre del punto de control, y se puede utilizar directamente con cualquier punto de control:

**TensorFlow/Keras:**

Al igual que `TFAutoModel`, la clase `AutoTokenizer` tomará la clase de tokenizador adecuada en la librería basada en el nombre del punto de control, y se puede utilizar directamente con cualquier punto de control:


```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Ahora podemos utilizar el tokenizador como se muestra en la sección anterior:

```python
tokenizer("Using a Transformer network is simple")
```

```python out
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Guardar un tokenizador es idéntico a guardar un modelo:

```py
tokenizer.save_pretrained("directorio_en_mi_computador")
```

Hablaremos más sobre `token_type_ids` en el [Capítulo 3](/course/chapter3), y explicaremos la clave `attention_mask` un poco más tarde. Primero, veamos cómo se generan los `input_ids`. Para ello, tendremos que ver los métodos intermedios del tokenizador.

## Encoding


**Video:** [Ver en YouTube](https://youtu.be/Yffk5aydLzg)


La traducción de texto a números se conoce como _codificación_. La codificación se realiza en un proceso de dos pasos: la tokenización, seguida de la conversión a IDs de entrada.

Como hemos visto, el primer paso es dividir el texto en palabras (o partes de palabras, símbolos de puntuación, etc.), normalmente llamadas *tokens*. Hay múltiples reglas que pueden gobernar ese proceso, por lo que necesitamos instanciar el tokenizador usando el nombre del modelo, para asegurarnos de que usamos las mismas reglas que se usaron cuando se preentrenó el modelo.

El segundo paso es convertir esos tokens en números, para poder construir un tensor con ellos y alimentar el modelo. Para ello, el tokenizador tiene un *vocabulario*, que es la parte que descargamos cuando lo instanciamos con el método `from_pretrained()`. De nuevo, necesitamos usar el mismo vocabulario que se usó cuando el modelo fue preentrenado.

Para entender mejor los dos pasos, los exploraremos por separado. Ten en cuenta que utilizaremos algunos métodos que realizan partes del proceso de tokenización por separado para mostrarte los resultados intermedios de esos pasos, pero en la práctica, deberías llamar al tokenizador directamente en tus _inputs_ (como se muestra en la sección 2).

### Tokenization

El proceso de tokenización se realiza mediante el método `tokenize()` del tokenizador:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

La salida de este método es una lista de cadenas, o tokens:

```python out
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```

Este tokenizador es un tokenizador de subpalabras: divide las palabras hasta obtener tokens que puedan ser representados por su vocabulario. Este es el caso de `transformer`, que se divide en dos tokens: `transform` y `##er`.

### De tokens a IDs de entrada

La conversión a IDs de entrada se hace con el método del tokenizador `convert_tokens_to_ids()`:

```py
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```python out
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```

Estos resultados, una vez convertidos en el tensor del marco apropiado, pueden utilizarse como entradas de un modelo, como se ha visto anteriormente en este capítulo.

> [!TIP]
> ✏️ **Try it out!** Replica los dos últimos pasos (tokenización y conversión a IDs de entrada) en las frases de entrada que utilizamos en la sección 2 ("Llevo toda la vida esperando un curso de HuggingFace" y "¡Odio tanto esto!"). Comprueba que obtienes los mismos ID de entrada que obtuvimos antes!

## Decodificación

La *decodificación* va al revés: a partir de los índices del vocabulario, queremos obtener una cadena. Esto se puede hacer con el método `decode()` de la siguiente manera:

```py
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

```python out
'Using a Transformer network is simple'
```

Notemos que el método `decode` no sólo convierte los índices de nuevo en tokens, sino que también agrupa los tokens que formaban parte de las mismas palabras para producir una frase legible. Este comportamiento será extremadamente útil cuando utilicemos modelos que predigan texto nuevo (ya sea texto generado a partir de una indicación, o para problemas de secuencia a secuencia como la traducción o el resumen).

A estas alturas deberías entender las operaciones atómicas que un tokenizador puede manejar: tokenización, conversión a IDs, y conversión de IDs de vuelta a una cadena. Sin embargo, sólo hemos rozado la punta del iceberg. En la siguiente sección, llevaremos nuestro enfoque a sus límites y echaremos un vistazo a cómo superarlos.


---



# Manejando Secuencias Múltiples


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/M6adb1j2jPI)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/ROxrFOEbsQE)


En la sección anterior, hemos explorado el caso de uso más sencillo: hacer inferencia sobre una única secuencia de poca longitud. Sin embargo, surgen algunas preguntas:

- ¿Cómo manejamos las secuencias múltiples?
- ¿Cómo manejamos las secuencias múltiples *de diferentes longitudes*?
- ¿Son los índices de vocabulario las únicas entradas que permiten que un modelo funcione bien?
- ¿Existe una secuencia demasiado larga?

Veamos qué tipo de problemas plantean estas preguntas, y cómo podemos resolverlos utilizando la API de Transformers 🤗.

## Los modelos esperan Baches de entrada 

En el ejercicio anterior has visto cómo las secuencias se traducen en listas de números. Convirtamos esta lista de números en un tensor y enviémoslo al modelo:


**PyTorch:**

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# Esta línea va a fallar:
model(input_ids)
```

```python out
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

**TensorFlow/Keras:**

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tf.constant(ids)
# Esta línea va a fallar:
model(input_ids)
```

```py out
InvalidArgumentError: Input to reshape is a tensor with 14 values, but the requested shape has 196 [Op:Reshape]
```

¡Oh, no! ¿Por qué ha fallado esto? "Hemos seguido los pasos de la tubería en la sección 2.

El problema es que enviamos una sola secuencia al modelo, mientras que los modelos de 🤗 Transformers esperan múltiples frases por defecto. Aquí tratamos de hacer todo lo que el tokenizador hizo detrás de escena cuando lo aplicamos a una `secuencia`, pero si te fijas bien, verás que no sólo convirtió la lista de IDs de entrada en un tensor, sino que le agregó una dimensión encima:


**PyTorch:**

```py
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

```python out
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
```

**TensorFlow/Keras:**

```py
tokenized_inputs = tokenizer(sequence, return_tensors="tf")
print(tokenized_inputs["input_ids"])
```

```py out
<tf.Tensor: shape=(1, 16), dtype=int32, numpy=
array([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,
        12172,  2607,  2026,  2878,  2166,  1012,   102]], dtype=int32)>
```

Intentémoslo de nuevo y añadamos una nueva dimensión encima:


**PyTorch:**

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

**TensorFlow/Keras:**

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = tf.constant([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

Imprimimos los IDs de entrada así como los logits resultantes - aquí está la salida:


**PyTorch:**

```python out
Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]
```

**TensorFlow/Keras:**

```py out
Input IDs: tf.Tensor(
[[ 1045  1005  2310  2042  3403  2005  1037 17662 12172  2607  2026  2878
   2166  1012]], shape=(1, 14), dtype=int32)
Logits: tf.Tensor([[-2.7276208  2.8789377]], shape=(1, 2), dtype=float32)
```

*El "batching"* es el acto de enviar varias frases a través del modelo, todas a la vez. Si sólo tienes una frase, puedes construir un lote con una sola secuencia: 

```
batched_ids = [ids, ids]
```

Se trata de un lote de dos secuencias idénticas.

> [!TIP]
> ✏️ **Try it out!** Convierte esta lista `batched_ids` en un tensor y pásalo por tu modelo. Comprueba que obtienes los mismos logits que antes (¡pero dos veces!).

La creación de lotes permite que el modelo funcione cuando lo alimentas con múltiples sentencias. Utilizar varias secuencias es tan sencillo como crear un lote con una sola secuencia. Sin embargo, hay un segundo problema. Cuando se trata de agrupar dos (o más) frases, éstas pueden ser de diferente longitud. Si alguna vez ha trabajado con tensores, sabrá que deben tener forma rectangular, por lo que no podrá convertir la lista de IDs de entrada en un tensor directamente. Para evitar este problema, usamos el *padding* para las entradas.

## Padding a las entradas

La siguiente lista de listas no se puede convertir en un tensor:

```py no-format
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```
Para solucionar esto, utilizaremos *padding* para que nuestros tensores tengan una forma rectangular. El acolchado asegura que todas nuestras sentencias tengan la misma longitud añadiendo una palabra especial llamada *padding token* a las sentencias con menos valores. Por ejemplo, si tienes 10 frases con 10 palabras y 1 frase con 20 palabras, el relleno asegurará que todas las frases tengan 20 palabras. En nuestro ejemplo, el tensor resultante tiene este aspecto:

```py no-format
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```
El ID del *padding token* se puede encontrar en `tokenizer.pad_token_id`. Usémoslo y enviemos nuestras dos sentencias a través del modelo de forma individual y por lotes:


**PyTorch:**

```py no-format
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

```python out
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
```

**TensorFlow/Keras:**

```py no-format
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(tf.constant(sequence1_ids)).logits)
print(model(tf.constant(sequence2_ids)).logits)
print(model(tf.constant(batched_ids)).logits)
```

```py out
tf.Tensor([[ 1.5693678 -1.3894581]], shape=(1, 2), dtype=float32)
tf.Tensor([[ 0.5803005  -0.41252428]], shape=(1, 2), dtype=float32)
tf.Tensor(
[[ 1.5693681 -1.3894582]
 [ 1.3373486 -1.2163193]], shape=(2, 2), dtype=float32)
```


Hay un problema con los logits en nuestras predicciones por lotes: la segunda fila debería ser la misma que los logits de la segunda frase, ¡pero tenemos valores completamente diferentes!

Esto se debe a que la característica clave de los modelos Transformer son las capas de atención que *contextualizan* cada token. Éstas tendrán en cuenta los tokens de relleno, ya que atienden a todos los tokens de una secuencia. Para obtener el mismo resultado al pasar oraciones individuales de diferente longitud por el modelo o al pasar un lote con las mismas oraciones y el padding aplicado, tenemos que decirles a esas capas de atención que ignoren los tokens de padding. Esto se hace utilizando una máscara de atención.

## Máscaras de atención

*Las máscaras de atención* son tensores con la misma forma que el tensor de IDs de entrada, rellenados con 0s y 1s: los 1s indican que los tokens correspondientes deben ser atendidos, y los 0s indican que los tokens correspondientes no deben ser atendidos (es decir, deben ser ignorados por las capas de atención del modelo).


**PyTorch:**

```py no-format
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

```python out
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```

**TensorFlow/Keras:**

```py no-format
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(tf.constant(batched_ids), attention_mask=tf.constant(attention_mask))
print(outputs.logits)
```

```py out
tf.Tensor(
[[ 1.5693681  -1.3894582 ]
 [ 0.5803021  -0.41252586]], shape=(2, 2), dtype=float32)
```


Ahora obtenemos los mismos logits para la segunda frase del lote.

Podemos ver que el último valor de la segunda secuencia es un ID de relleno, que es un valor 0 en la máscara de atención.

> [!TIP]
> ✏️ **Try it out!** Aplique la tokenización manualmente a las dos frases utilizadas en la sección 2 ("Llevo toda la vida esperando un curso de HuggingFace" y "¡Odio tanto esto!"). Páselas por el modelo y compruebe que obtiene los mismos logits que en la sección 2. Ahora júntalos usando el token de relleno, y luego crea la máscara de atención adecuada. Comprueba que obtienes los mismos resultados al pasarlos por el modelo.

## Secuencias largas

Con los modelos Transformer, hay un límite en la longitud de las secuencias que podemos pasar a los modelos. La mayoría de los modelos manejan secuencias de hasta 512 o 1024 tokens, y se bloquean cuando se les pide que procesen secuencias más largas. Hay dos soluciones a este problema:

- Usar un modelo que soporte secuencias largas
- Truncar tus secuencias

Los modelos tienen diferentes longitudes de secuencia soportadas, y algunos se especializan en el manejo de secuencias muy largas. Un ejemplo es [Longformer](https://huggingface.co/transformers/model_doc/longformer.html) y otro es [LED](https://huggingface.co/transformers/model_doc/led.html). Si estás trabajando en una tarea que requiere secuencias muy largas, te recomendamos que eches un vistazo a esos modelos.

En caso contrario, le recomendamos que trunque sus secuencias especificando el parámetro `max_sequence_length`:

```py
sequence = sequence[:max_sequence_length]
```


---



# Poniendo todo junto


En las últimas secciones, hemos hecho nuestro mejor esfuerzo para realizar la mayor parte del trabajo a mano. Exploramos como funcionan los tokenizadores y vimos la tokenización, conversión a IDs de entrada, relleno, truncado, y máscaras de atención.

Sin embargo, como vimos en la sección 3, la API de transformadores 🤗 puede manejar todo esto por nosotros con una función de alto nivel la cual trataremos aquí. Cuando llamas a tu `tokenizer` directamente en una sentencia, obtienes entradas que están lista para pasar a tu modelo:

```py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

Aquí la variable `model_inputs` contiene todo lo necesario para que un modelo opere bien. Para DistilBERT, que incluye los IDs de entrada también como la máscara de atención. Otros modelos que aceptan entradas adicionales también tendrán las salidas del objeto `tokenizer`.

Como veremos en los ejemplos de abajo, este método es muy poderoso. Primero, puede tokenizar una sola secuencia:

```py
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

También maneja múltiples secuencias a la vez, sin cambios en la API:

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

Puede rellenar de acuerdo a varios objetivos:

```py
# Rellenar las secuencias hasta la mayor longitud de secuencia
model_inputs = tokenizer(sequences, padding="longest")

# Rellenar las secuencias hasta la máxima longitud del modelo
# (512 para BERT o DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Rellenar las secuencias hasta la máxima longitud especificada
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

También puede truncar secuencias:

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Truncar las secuencias más largas que la máxima longitud del modelo
# (512 para BERT o DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Truncar las secuencias más largas que la longitud especificada
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

El objeto `tokenizer` puede manejar la conversión a tensores de frameworks específicos, los cuales pueden ser enviados directamente al modelo. Por ejemplo, en el siguiente código de ejemplo estamos solicitando al tokenizer que regrese los tensores de los distintos frameworks — `"pt"` regresa tensores de PyTorch, `"tf"` regresa tensores de TensorFlow, y `"np"` regresa arreglos de NumPy: 

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Devuelve tensores PyTorch
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Devuelve tensores TensorFlow
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Devuelve arrays Numpy
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Tokens especiales

Si damos un vistazo a los IDs de entrada retornados por el tokenizer, veremos que son un poquito diferentes a lo que teníamos anteriormente:

```py
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

```python out
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```

Se agregó un ID de token al principio, y uno al final. Decodifiquemos las dos secuencias de IDs de arriba para ver de que se trata:

```py
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

```python out
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
"i've been waiting for a huggingface course my whole life."
```

El tokenizador agregó la palabra especial `[CLS]` al principio y la palabra especial `[SEP]` al final. Esto se debe a que el modelo fue preentrenado con esos, así para obtener los mismos resultados por inferencia necesitamos agregarlos también. Nota que algunos modelos no agregan palabras especiales, o agregan unas distintas; los modelos también pueden agregar estas palabras especiales sólo al principio, o sólo al final. En cualquier caso, el tokenizador sabe cuáles son las esperadas y se encargará de ello por tí.

## Conclusión: Del tokenizador al modelo

Ahora que hemos visto todos los pasos individuales que el objeto `tokenizer` usa cuando se aplica a textos, veamos una última vez cómo maneja varias secuencias (¡relleno!), secuencias muy largas (¡truncado!), y múltiples tipos de tensores con su API principal:


**PyTorch:**

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

**TensorFlow/Keras:**

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)
```



---

# ¡Has completado el uso básico!


¡Buen trabajo siguiendo el curso hasta ahora! Para recapitular, en este capítulo tú:

- Aprendiste los bloques de construcción básicos de un modelo Transformer.
- Aprendiste lo que compone a un pipeline de tokenización.
- Viste cómo usar un modelo Transformer en la práctica.
- Aprendiste cómo aprovechar un tokenizador para convertir texto a tensores que sean entendibles por el modelo.
- Configuraste un tokenizador y un modelo juntos para pasar dle texto a predicciones.
- Aprendiste las limitaciones de los IDs de entrada, y aprendiste acerca de máscaras de atención.
- Jugaste con los métodos del tokenizador versátiles y configurables.

A partir de ahora, serás capaz de navegar libremente por la documentación de 🤗 Transformers: el vocabulario te sonará familiar, ya que has visto los métodos que usarás la mayor parte del tiempo.


---




# Quiz de final de capítulo


### 1. ¿Cuál es el orden del pipeline de modelado del lenguaje?


- Primero, el modelo que maneja el texto y devuelve las peticiones sin procesar. El tokenizador luego da sentido a estas predicciones y las convierte nuevamente en texto cuando es necesario.
- Primero, el tokenizador, que maneja el texto y regresa IDs. El modelo maneja estos IDs y produce una predicción, la cual puede ser algún texto.
- El tokenizador maneja texto y regresa IDs. El modelo maneja estos IDs y produce una predicción. El tokenizador puede luego ser usado de nuevo para convertir estas predicciones de vuelta a texto.


### 2. ¿Cuántas dimensiones tiene el tensor producido por el modelo base de Transformer y cuáles son?


- 1: La longitud de secuencia y el tamaño del lote
- 2: La longitud de secuencia y el tamaño oculto
- 3: La longitud de secuencia, el tamaño de lote y el tamaño oculto


### 3. ¿Cuál de los siguientes es un ejemplo de tokenización de subpalabras?


- WordPiece
- Tokenización basada en caracteres
- División por espacios en blanco y puntuación
- BPE
- Unigrama
- Ninguno de los anteriores


### 4. ¿Qué es una cabeza del modelo?


- Un componente de la red de Transformer base que redirecciona los tensores a sus capas correctas
- También conocido como el mecanismo de autoatención, adapta la representación de un token de acuerdo a los otros tokens de la secuencia
- Un componente adicional, compuesto usualmente de una o unas pocas capas, para convertir las predicciones del transformador a una salida específica de la tarea


**PyTorch:**

### 5. ¿Qué es un AutoModel?


- Un modelo que entrena automáticamente en tus datos
- Un objeto que devuelve la arquitectura correcta basado en el punto de control
- Un modelo que detecta automáticamente el lenguaje usado por sus entradas para cargar los pesos correctos


**TensorFlow/Keras:**

### 5. ¿Qué es un TFAutoModel?


- Un modelo que entrena automáticamente en tus datos
- Un objeto que devuelve la arquitectura correcta basado en el punto de control
- Un modelo que detecta automáticamente el lenguaje usado por sus entradas para cargar los pesos correctos


### 6. ¿Cuáles son las técnicas a tener en cuenta al realizar batching de secuencias de diferentes longitudes juntas?


- Truncado
- Returning tensors
- Relleno
- Enmascarado de atención


### 7. ¿Cuál es el punto de aplicar una función SoftMax a las salidas logits por un modelo de clasificación de secuencias?


- Suaviza los logits para que sean más fiables.
- Aplica un límite inferior y superior de modo que sean comprensibles.
- La suma total de la salida es entonces 1, dando como resultado una posible interpretación probabilística.


### 8. ¿En qué método se centra la mayor parte de la API del tokenizador?


- `encode`, ya que puede codificar texto en IDs e IDs en predicciones
- Llamar al objeto tokenizador directamente.
- `pad`
- `tokenize`


### 9. ¿Qué contiene la variable `result` en este código de ejemplo?

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
```


- Una lista de strings, cada string es un token
- Una lista de IDs
- Una cadena que contiene todos los tokens


**PyTorch:**

### 10. ¿Hay algo mal con el siguiente código?

```py
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```


- No, parece correcto.
- El tokenizador y el modelo siempre deben ser del mismo punto de control.
- Es una buena práctica rellenar y truncar con el tokenizador ya que cada entrada es un lote.


**TensorFlow/Keras:**

### 10. ¿Hay algo mal con el siguiente código?

```py
from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = TFAutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```


- No, parece correcto.
- El tokenizador y el modelo siempre deben ser del mismo punto de control.
- Es una buena práctica rellenar y truncar con el tokenizador ya que cada entrada es un lote.




---




# Cuestionario de fin de capitulo[[end-of-chapter-quiz]]


### 1. Cual es el orden del pipeline de modelado de lenguaje?


- Primero, el modelo, que maneja texto y devuelve predicciones crudas. El tokenizador luego da sentido a estas predicciones y las convierte de vuelta a texto cuando es necesario.
- Primero, el tokenizador, que maneja texto y devuelve IDs. El modelo maneja estos IDs y produce una prediccion, que puede ser algun texto.
- El tokenizador maneja texto y devuelve IDs. El modelo maneja estos IDs y produce una prediccion. El tokenizador puede entonces usarse nuevamente para convertir estas predicciones de vuelta a texto.


### 2. Cuantas dimensiones tiene el tensor de salida del modelo Transformer base, y cuales son?


- 2: La longitud de secuencia y el tamano del lote
- 2: La longitud de secuencia y el tamano oculto
- 3: La longitud de secuencia, el tamano del lote y el tamano oculto


### 3. Cual de los siguientes es un ejemplo de tokenizacion por subpalabras?


- WordPiece
- Tokenizacion basada en caracteres
- Division por espacios en blanco y puntuacion
- BPE
- Unigram
- Ninguna de las anteriores


### 4. Que es una cabeza de modelo?


- Un componente de la red Transformer base que redirige tensores a sus capas correctas
- Tambien conocido como el mecanismo de auto-atencion, adapta la representacion de un token segun los otros tokens de la secuencia
- Un componente adicional, usualmente compuesto de una o pocas capas, para convertir las predicciones del transformer en una salida especifica de la tarea


### 5. Que es un AutoModel?


- Un modelo que entrena automaticamente con tus datos
- Un objeto que devuelve la arquitectura correcta basandose en el checkpoint
- Un modelo que detecta automaticamente el idioma usado en sus entradas para cargar los pesos correctos


### 6. Cuales son las tecnicas a tener en cuenta al agrupar secuencias de diferentes longitudes juntas?


- Truncamiento
- Devolver tensores
- Relleno (Padding)
- Enmascaramiento de atencion


### 7. Cual es el proposito de aplicar una funcion SoftMax a los logits producidos por un modelo de clasificacion de secuencias?


- Suaviza los logits para que sean mas confiables.
- Aplica un limite inferior y superior para que sean comprensibles.
- La suma total de la salida es entonces 1, resultando en una posible interpretacion probabilistica.


### 8. Alrededor de que metodo se centra la mayor parte de la API del tokenizador?


- `encode`, ya que puede codificar texto en IDs e IDs en predicciones
- Llamando al objeto tokenizador directamente.
- `pad`
- `tokenize`


### 9. Que contiene la variable `result` en este ejemplo de codigo?

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
```


- Una lista de cadenas, cada cadena siendo un token
- Una lista de IDs
- Una cadena conteniendo todos los tokens


### 10. Hay algo mal con el siguiente codigo?

```py
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```


- No, parece correcto.
- El tokenizador y el modelo siempre deben ser del mismo checkpoint.
- Es buena practica rellenar y truncar con el tokenizador ya que cada entrada es un lote.



---

# 3. Ajuste fino de un modelo preentrenado



# Introducción


En el [Capítulo 2](/course/chapter2) exploramos cómo usar los tokenizadores y modelos preentrenados para realizar predicciones. Pero, ¿qué pasa si deseas ajustar un modelo preentrenado con tu propio conjunto de datos?


**PyTorch:**

* Cómo preparar un conjunto de datos grande desde el Hub.
* Cómo usar la API de alto nivel del entrenador para ajustar un modelo.
* Cómo usar un bucle personalizado de entrenamiento.
* Cómo aprovechar la librería 🤗 Accelerate para fácilmente ejecutar el bucle personalizado de entrenamiento en cualquier configuración distribuida.

**TensorFlow/Keras:**

* Cómo preparar un conjunto de datos grande desde el Hub.
* Cómo usar Keras para ajustar un modelo.
* Cómo usar Keras para obtener predicciones.
* Cómo usar una métrica personalizada.


Para subir tus puntos de control (*checkpoints*) en el Hub de Hugging Face, necesitas una cuenta en huggingface.co: [crea una cuenta](https://huggingface.co/join)

---



# Procesando los datos


**PyTorch:**

Continuando con el ejemplo del [capítulo anterior](/course/chapter2), aquí mostraremos como podríamos entrenar un clasificador de oraciones/sentencias en PyTorch.:

```python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

**TensorFlow/Keras:**

Continuando con el ejemplo del [capítulo anterior](/course/chapter2), aquí mostraremos como podríamos entrenar un clasificador de oraciones/sentencias en TensorFlow:

```python
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Igual que antes
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = dict(tokenizer(sequences, padding=True, truncation=True, return_tensors="tf"))

# Esto es nuevo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
labels = tf.convert_to_tensor([1, 1])
model.train_on_batch(batch, labels)
```


Por supuesto, entrenando el modelo con solo dos oraciones no va a producir muy buenos resultados. Para obtener mejores resultados, debes preparar un conjunto de datos más grande.

En esta sección usaremos como ejemplo el conjunto de datos MRPC (Cuerpo de paráfrasis de investigaciones de Microsoft), que fue presentado en el [artículo](https://www.aclweb.org/anthology/I05-5002.pdf) de William B. Dolan and Chris Brockett. El conjunto de datos consiste en 5,801 pares of oraciones, con una etiqueta que indica si son paráfrasis o no. (es decir, si ambas oraciones significan lo mismo). Hemos seleccionado el mismo para este capítulo porque es un conjunto de datos pequeño que facilita la experimentación y entrenamiento sobre él.

### Cargando un conjunto de datos desde el Hub


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/_BZearw7f0w)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/W_gMJF0xomE)


El Hub no solo contiene modelos; sino que también tiene múltiples conjunto de datos en diferentes idiomas. Puedes explorar los conjuntos de datos [aquí](https://huggingface.co/datasets), y recomendamos que trates de cargar y procesar un nuevo conjunto de datos una vez que hayas revisado esta sección (mira la documentación general [aquí](https://huggingface.co/docs/datasets/loading)). Por ahora, enfoquémonos en el conjunto de datos MRPC! Este es uno de los 10 conjuntos de datos que comprende el [punto de referencia GLUE](https://gluebenchmark.com/), el cual es un punto de referencia académico que se usa para medir el desempeño de modelos ML sobre 10 tareas de clasificación de texto.

La librería 🤗 Datasets provee un comando muy simple para descargar y memorizar un conjunto de datos en el Hub. Podemos descargar el conjunto de datos de la siguiente manera:

> [!TIP]
> ⚠️ **Advertencia** Asegúrate de que `datasets` esté instalado ejecutando `pip install datasets`. Luego, carga el conjunto de datos MRPC y imprímelo para ver qué contiene. 

```py
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

Como puedes ver, obtenemos un objeto `DatasetDict` que contiene los conjuntos de datos de entrenamiento, de validación y de pruebas. Cada uno de estos contiene varias columnas (`sentence1`, `sentence2`, `label`, and `idx`) y un número variable de filas, que son el número de elementos en cada conjunto (asi, que hay 3,668 pares de oraciones en el conjunto de entrenamiento, 408 en el de validación, y 1,725 en el pruebas)

Este comando descarga y almacena el conjunto de datos, por defecto en *~/.cache/huggingface/dataset*. Recuerda del Capítulo 2 que puedes personalizar tu carpeta mediante la configuración de la variable de entorno `HF_HOME`.

Podemos acceder a cada par de oraciones en nuestro objeto `raw_datasets` usando indexación, como con un diccionario.

```py
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

```python out
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```

Podemos ver que las etiquetas ya son números enteros, así que no es necesario hacer ningún preprocesamiento. Para saber cual valor corresponde con cual etiqueta, podemos inspeccionar el atributo `features` de nuestro `raw_train_dataset`. Esto indicara el tipo dato de cada columna:

```py
raw_train_dataset.features
```

```python out
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```

Internamente, `label` es del tipo de dato `ClassLabel`, y la asociación de valores enteros y sus etiquetas esta almacenado en la carpeta *names*. `0` corresponde con `not_equivalent`, y `1` corresponde con `equivalent`.

> [!TIP]
> ✏️ **¡Inténtalo!** Mira el elemento 15 del conjunto de datos de entrenamiento y el elemento 87 del conjunto de datos de validación. Cuáles son sus etiquetas?

### Preprocesando un conjunto de datos


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/0u3ioSwev3s)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/P-rZWqcB6CE)


Para preprocesar el conjunto de datos, necesitamos convertir el texto en números que puedan ser entendidos por el modelo. Como viste en el [capítulo anterior](/course/chapter2), esto se hace con el tokenizador. Podemos darle al tokenizador una oración o una lista de oraciones, así podemos tokenizar directamente todas las primeras y las segundas oraciones de cada par de la siguiente manera:

```py
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

Sin embargo, no podemos simplemente pasar dos secuencias al modelo y obtener una predicción indicando si estas son paráfrasis o no. Necesitamos manipular las dos secuencias como un par y aplicar el preprocesamiento apropiado.
Afortunadamente, el tokenizador puede recibir también un par de oraciones y preparar las misma de una forma que nuestro modelo BERT espera:

```py
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python out
{
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

Nosotros consideramos las llaves `input_ids` y `attention_mask` en el [Capítulo 2](/course/chapter2), pero postergamos hablar sobre la llave `token_type_ids`. En este ejemplo, esta es la que le dice al modelo cual parte de la entrada es la primera oración y cual es la segunda.

> [!TIP]
> ✏️ **¡Inténtalo!** Toma el elemento 15 del conjunto de datos de entrenamiento y tokeniza las dos oraciones independientemente y como un par. Cuál es la diferencia entre los dos resultados?

Si convertimos los IDs dentro de `input_ids` en palabras:

```py
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```

obtendremos:

```python out
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```

De esta manera vemos que el modelo espera las entradas de la siguiente forma `[CLS] sentence1 [SEP] sentence2 [SEP]` cuando hay dos oraciones. Alineando esto con los `token_type_ids` obtenemos:

```python out
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

Como puedes observar, las partes de la entrada que corresponden a `[CLS] sentence1 [SEP]` todas tienen un tipo de token ID `0`, mientras que las otras partes que corresponden a `sentence2 [SEP]`, todas tienen tipo ID `1`.

Nótese que si seleccionas un punto de control diferente, no necesariamente tendrás el `token_type_ids` en tus entradas tokenizadas (por ejemplo, ellas no aparecen si usas un modelo DistilBERT). Estas aparecen cuando el modelo sabe que hacer con ellas, porque las ha visto durante su etapa de preentrenamiento.

Aquí, BERT está preentrenado con tokens de tipo ID, y además del objetivo de modelado de lenguaje oculto que mencionamos en el [Capítulo 1](/course/chapter1), también tiene el objetivo llamado _predicción de la siguiente oración_. El objetivo con esta tarea es modelar la relación entre pares de oraciones.

Para predecir la siguiente oración, el modelo recibe pares de oraciones (con tokens ocultados aleatoriamente) y se le pide que prediga si la segunda secuencia sigue a la primera. Para que la tarea no sea tan simple, la mitad de las veces las oraciones están seguidas en el texto original de donde se obtuvieron, y la otra mitad las oraciones vienen de dos documentos distintos.

En general, no debes preocuparte si los `token_type_ids` están o no en las entradas tokenizadas: con tal de que uses el mismo punto de control para el tokenizador y el modelo, todo estará bien porque el tokenizador sabe qué pasarle a su modelo.

Ahora que hemos visto como nuestro tokenizador puede trabajar con un par de oraciones, podemos usarlo para tokenizar todo el conjunto de datos: como en el [capítulo anterior](/course/es/chapter2), podemos darle al tokenizador una lista de pares de oraciones, dándole la lista de las primeras oraciones, y luego la lista de las segundas oraciones. Esto también es compatible con las opciones de relleno y truncamiento que vimos en el [Capítulo 2](/course/chapter2). Por lo tanto, una manera de preprocesar el conjunto de datos de entrenamiento sería:

```py
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

Esto funciona bien, pero tiene la desventaja de que devuelve un diccionario (con nuestras llaves, `input_ids`, `attention_mask`, and `token_type_ids`, y valores que son listas de listas). Además va a trabajar solo si tienes suficiente memoria principal para almacenar todo el conjunto de datos durante la tokenización (mientras que los conjuntos de datos de la librería 🤗 Datasets son archivos [Apache Arrow](https://arrow.apache.org/) almacenados en disco, y así solo mantienes en memoria las muestras que necesitas).

Para mantener los datos como un conjunto de datos, usaremos el método [`Dataset.map()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map). Este también nos ofrece una flexibilidad adicional en caso de que necesitemos preprocesamiento mas allá de la tokenización. El método `map()` trabaja aplicando una función sobre cada elemento del conjunto de datos, así que definamos una función para tokenizar nuestras entradas:

```py
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

Esta función recibe un diccionario (como los elementos de nuestro conjunto de datos) y devuelve un nuevo diccionario con las llaves `input_ids`, `attention_mask`, y `token_type_ids`. Nótese que también funciona si el diccionario `example` contiene múltiples elementos (cada llave con una lista de oraciones) debido a que el `tokenizador` funciona con listas de pares de oraciones, como se vio anteriormente. Esto nos va a permitir usar la opción `batched=True` en nuestra llamada a `map()`, lo que acelera la tokenización significativamente. El `tokenizador` es respaldado por un tokenizador escrito en Rust que viene de la librería [🤗 Tokenizers](https://github.com/huggingface/tokenizers). Este tokenizador puede ser muy rápido, pero solo si le da muchas entradas al mismo tiempo.

Nótese que por ahora hemos dejado el argumento `padding` fuera de nuestra función de tokenización. Esto es porque rellenar todos los elementos hasta su máxima longitud no es eficiente: es mejor rellenar los elementos cuando se esta construyendo el lote, debido a que solo debemos rellenar hasta la máxima longitud en el lote, pero no en todo el conjunto de datos. Esto puede ahorrar mucho tiempo y poder de procesamiento cuando las entradas tienen longitudes variables.

Aquí se muestra como se aplica la función de tokenización a todo el conjunto de datos en un solo paso. Estamos usando `batched=True` en nuestra llamada a `map` para que la función sea aplicada a múltiples elementos de nuestro conjunto de datos al mismo tiempo, y no a cada elemento por separado. Esto permite un preprocesamiento más rápido.

```py
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

La manera en que la librería 🤗 Datasets aplica este procesamiento es a través de campos añadidos al conjunto de datos, uno por cada diccionario devuelto por la función de preprocesamiento.

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

Hasta puedes usar multiprocesamiento cuando aplicas la función de preprocesamiento con `map()` pasando el argumento `num_proc`. Nosotros no usamos esta opción porque los tokenizadores de la librería 🤗 Tokenizers usa múltiples hilos de procesamiento para tokenizar rápidamente nuestros elementos, pero sino estas usando un tokenizador rápido respaldado por esta librería, esta opción puede acelerar tu preprocesamiento.

Nuestra función `tokenize_function` devuelve un diccionario con las llaves `input_ids`, `attention_mask`, y `token_type_ids`, así que esos tres campos son adicionados a todas las divisiones de nuestro conjunto de datos. Nótese que pudimos haber cambiado los campos existentes si nuestra función de preprocesamiento hubiese devuelto un valor nuevo para cualquiera de las llaves en el conjunto de datos al que le aplicamos `map()`.

Lo último que necesitamos hacer es rellenar todos los elementos hasta la longitud del elemento más largo al momento de agrupar los elementos - a esta técnica la llamamos *relleno dinámico*.

### Relleno Dinámico


**Video:** [Ver en YouTube](https://youtu.be/7q5NyFT8REg)


**PyTorch:**

La función responsable de juntar los elementos dentro de un lote es llamada *función de cotejo*. Esta es un argumento que puedes pasar cuando construyes un `DataLoader`, cuya función por defecto convierte tus elementos a tensores PyTorch y los concatena (recursivamente si los elementos son listas, tuplas o diccionarios). Esto no será posible en nuestro caso debido a que las entradas que tenemos no tienen el mismo tamaño. Hemos pospuesto el relleno, para aplicarlo sólo cuando se necesita en cada lote y evitar tener entradas muy largas con mucho relleno. Esto va a acelerar el entrenamiento significativamente, pero nótese que esto puede causar problemas si estás entrenando en un TPU - Los TPUs prefieren tamaños fijos, aún cuando requieran relleno adicional.

**TensorFlow/Keras:**

La función responsable de juntar los elementos dentro de un lote es llamada *función de cotejo*. Esta es un argumento que puedes pasar cuando construyes un `DataLoader`, cuya función por defecto convierte tus elementos a un tf.Tensor y los concatena (recursivamente si los elementos son listas, tuplas o diccionarios). Esto no será posible en nuestro caso debido a que las entradas que tenemos no tienen el mismo tamaño. Hemos pospuesto el relleno, para aplicarlo sólo cuando se necesita en cada lote y evitar tener entradas muy largas con mucho relleno. Esto va a acelerar el entrenamiento significativamente, pero nótese que esto puede causar problemas si estás entrenando en un TPU - Los TPUs prefieren tamaños fijos, aún cuando requieran relleno adicional.


Para poner esto en práctica, tenemos que definir una función de cotejo que aplique la cantidad correcta de relleno a los elementos del conjunto de datos que queremos agrupar. Afortunadamente, la librería 🤗 Transformers nos provee esta función mediante `DataCollatorWithPadding`. Esta recibe un tokenizador cuando la creas (para saber cual token de relleno se debe usar, y si el modelo espera el relleno a la izquierda o la derecha en las entradas) y hace todo lo que necesitas:


**PyTorch:**

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

**TensorFlow/Keras:**

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```


Para probar este nuevo juguete, tomemos algunos elementos de nuestro conjunto de datos de entrenamiento para agruparlos. Aquí, removemos las columnas `idx`, `sentence1`, and `sentence2` ya que éstas no se necesitan y contienen cadenas (y no podemos crear tensores con cadenas), miremos las longitudes de cada elemento en el lote.

```py
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
```

```python out
[50, 59, 47, 67, 59, 50, 62, 32]
```

Como era de esperarse, obtenemos elementos de longitud variable, desde 32 hasta 67. El relleno dinámico significa que los elementos en este lote deben ser rellenos hasta una longitud de 67, que es la máxima longitud en el lote. Sin relleno dinámico, todos los elementos tendrían que haber sido rellenos hasta el máximo de todo el conjunto de datos, o el máximo aceptado por el modelo. Verifiquemos que nuestro `data_collator` esta rellenando dinámicamente el lote de la manera apropiada:

```py
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

{#if fw === 'tf'}

```python out
{'attention_mask': TensorShape([8, 67]),
 'input_ids': TensorShape([8, 67]),
 'token_type_ids': TensorShape([8, 67]),
 'labels': TensorShape([8])}
```

{:else}

```python out
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
```

¡Luce bien! Ahora que hemos convertido el texto crudo a lotes que nuestro modelo puede aceptar, estamos listos para ajustarlo!

{/if}

> [!TIP]
> ✏️ **¡Inténtalo!** Reproduce el preprocesamiento en el conjunto de datos GLUE SST-2. Es un poco diferente ya que esta compuesto de oraciones individuales en lugar de pares, pero el resto de lo que hicimos debería ser igual. Para un reto mayor, intenta escribir una función de preprocesamiento que trabaje con cualquiera de las tareas GLUE.

{#if fw === 'tf'}

Ahora que tenemos nuestro conjunto de datos y el cotejador de datos, necesitamos juntarlos. Nosotros podríamos cargar lotes de datos y cotejarlos, pero eso sería mucho trabajo, y probablemente no muy eficiente. En cambio, existe un método que ofrece una solución eficiente para este problema: `to_tf_dataset()`. Este envuelve un `tf.data.Dataset` alrededor de tu conjunto de datos, con una función opcional de cotejo. `tf.data.Dataset` es un formato nativo de TensorFlow que Keras puede usar con el `model.fit()`, así este método convierte inmediatamente un conjunto de datos 🤗 a un formato que viene listo para entrenamiento. Veámoslo en acción con nuestro conjunto de datos.

```py
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)
```

¡Y eso es todo! Ahora podemos usar esos conjuntos de datos en nuestra próxima clase, donde el entrenamiento será mas sencillo después de todo el trabajo de preprocesamiento de datos.

{/if}


---



# Ajuste de un modelo con la API Trainer


**Video:** [Ver en YouTube](https://youtu.be/nvBXf7s7vTI)


🤗 Transformers incluye una clase `Trainer` para ayudarte a ajustar cualquiera de los modelos preentrenados proporcionados en tu dataset. Una vez que hayas hecho todo el trabajo de preprocesamiento de datos de la última sección, sólo te quedan unos pocos pasos para definir el `Trainer`. La parte más difícil será preparar el entorno para ejecutar `Trainer.train()`, ya que se ejecutará muy lentamente en una CPU. Si no tienes una GPU preparada, puedes acceder a GPUs o TPUs gratuitas en [Google Colab](https://colab.research.google.com/).

Los siguientes ejemplos de código suponen que ya has ejecutado los ejemplos de la sección anterior. Aquí tienes un breve resumen de lo que necesitas:

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Entrenamiento

El primer paso antes de que podamos definir nuestro `Trainer` es definir una clase `TrainingArguments` que contendrá todos los hiperparámetros que el `Trainer` utilizará para el entrenamiento y la evaluación del modelo. El único argumento que tienes que proporcionar es el directorio donde se guardarán tanto el modelo entrenado como los puntos de control (checkpoints). Para los demás parámetros puedes dejar los valores por defecto, deberían funcionar bastante bien para un ajuste básico.

```py
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

> [!TIP]
> 💡 Si quieres subir automáticamente tu modelo al Hub durante el entrenamiento, incluye `push_to_hub=True` en `TrainingArguments`. Aprenderemos más sobre esto en el [Capítulo 4](/course/chapter4/3).

El segundo paso es definir nuestro modelo. Como en el [capítulo anterior](/course/chapter2), utilizaremos la clase `AutoModelForSequenceClassification`, con dos etiquetas:

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Observarás que, a diferencia del [Capítulo 2](/course/chapter2), aparece una advertencia después de instanciar este modelo preentrenado. Esto se debe a que BERT no ha sido preentrenado para la clasificación de pares de frases, por lo que la cabeza del modelo preentrenado se ha eliminado y en su lugar se ha añadido una nueva cabeza adecuada para la clasificación de secuencias. Las advertencias indican que algunos pesos no se han utilizado (los correspondientes a la cabeza de preentrenamiento eliminada) y que otros se han inicializado aleatoriamente (los correspondientes a la nueva cabeza). La advertencia concluye animándote a entrenar el modelo, que es exactamente lo que vamos a hacer ahora.

Una vez que tenemos nuestro modelo, podemos definir un `Trainer` pasándole todos los objetos construidos hasta ahora: el `model`, los `training_args`, los datasets de entrenamiento y validación, nuestro `data_collator`, y nuestro `tokenizer`:

```py
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Ten en cuenta que cuando pasas el `tokenizer` como hicimos aquí, el `data_collator` por defecto utilizado por el `Trainer` será un `DataCollatorWithPadding` como definimos anteriormente, por lo que puedes omitir la línea `data_collator=data_collator`. De todas formas, era importante mostrarte esta parte del proceso en la sección 2.

Para ajustar el modelo en nuestro dataset, sólo tenemos que llamar al método `train()` de nuestro `Trainer`:

```py
trainer.train()
```

Esto iniciará el ajuste (que debería tardar un par de minutos en una GPU) e informará de la training loss cada 500 pasos. Sin embargo, no te dirá lo bien (o mal) que está rindiendo tu modelo. Esto se debe a que:

1. No le hemos dicho al `Trainer` que evalúe el modelo durante el entrenamiento especificando un valor para `evaluation_strategy`: `steps` (evaluar cada `eval_steps`) o `epoch` (evaluar al final de cada época).
2. No hemos proporcionado al `Trainer` una función `compute_metrics()` para calcular una métrica durante dicha evaluación (de lo contrario, la evaluación sólo habría impreso la pérdida, que no es un número muy intuitivo).

### Evaluación

Veamos cómo podemos construir una buena función `compute_metrics()` para utilizarla la próxima vez que entrenemos. La función debe tomar un objeto `EvalPrediction` (que es una tupla nombrada con un campo `predictions` y un campo `label_ids`) y devolverá un diccionario que asigna cadenas a flotantes (las cadenas son los nombres de las métricas devueltas, y los flotantes sus valores). Para obtener algunas predicciones de nuestro modelo, podemos utilizar el comando `Trainer.predict()`:

```py
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```python out
(408, 2) (408,)
```

La salida del método `predict()` es otra tupla con tres campos: `predictions`, `label_ids`, y `metrics`. El campo `metrics` sólo contendrá la pérdida en el dataset proporcionado, así como algunas métricas de tiempo (cuánto se tardó en predecir, en total y de media). Una vez que completemos nuestra función `compute_metrics()` y la pasemos al `Trainer`, ese campo también contendrá las métricas devueltas por `compute_metrics()`.

Como puedes ver, `predictions` es una matriz bidimensional con forma 408 x 2 (408 es el número de elementos del dataset que hemos utilizado). Esos son los logits de cada elemento del dataset que proporcionamos a `predict()` (como viste en el [capítulo anterior](/curso/capítulo2), todos los modelos Transformer devuelven logits). Para convertirlos en predicciones que podamos comparar con nuestras etiquetas, necesitamos tomar el índice con el valor máximo en el segundo eje:

```py
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

Ahora podemos comparar esas predicciones `preds` con las etiquetas. Para construir nuestra función `compute_metric()`, nos basaremos en las métricas de la librería 🤗 [Evaluate](https://github.com/huggingface/evaluate/). Podemos cargar las métricas asociadas al dataset MRPC tan fácilmente como cargamos el dataset, esta vez con la función `evaluate.load()`. El objeto devuelto tiene un método `compute()` que podemos utilizar para calcular de la métrica:

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

```python out
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```

Los resultados exactos que obtengas pueden variar, ya que la inicialización aleatoria de la cabeza del modelo podría cambiar las métricas obtenidas. Aquí, podemos ver que nuestro modelo tiene una precisión del 85,78% en el conjunto de validación y una puntuación F1 de 89,97. Estas son las dos métricas utilizadas para evaluar los resultados en el dataset MRPC para la prueba GLUE. La tabla del [paper de BERT](https://arxiv.org/pdf/1810.04805.pdf) recoge una puntuación F1 de 88,9 para el modelo base. Se trataba del modelo "uncased" (el texto se reescribe en minúsculas antes de la tokenización), mientras que nosotros hemos utilizado el modelo "cased" (el texto se tokeniza sin reescribir), lo que explica el mejor resultado.

Juntándolo todo obtenemos nuestra función `compute_metrics()`:

```py
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

Y para ver cómo se utiliza para informar de las métricas al final de cada época, así es como definimos un nuevo `Trainer` con nuestra función `compute_metrics()`:

```py
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

Ten en cuenta que hemos creado un nuevo `TrainingArguments` con su `evaluation_strategy` configurado como `"epoch"` y un nuevo modelo. De lo contrario sólo estaríamos continuando el entrenamiento del modelo que ya habíamos entrenado. Para lanzar una nueva ejecución de entrenamiento, ejecutamos:

```py
trainer.train()
```

Esta vez, nos informará de la pérdida de validación y las métricas al final de cada época, además de la pérdida de entrenamiento. De nuevo, la puntuación exacta de precisión/F1 que alcances puede ser un poco diferente de la que encontramos nosotros, debido a la inicialización aleatoria del modelo, pero debería estar en el mismo rango.

El `Trainer` funciona en múltiples GPUs o TPUs y proporciona muchas opciones, como el entrenamiento de precisión mixta (usa `fp16 = True` en tus argumentos de entrenamiento). Repasaremos todo lo que ofrece en el capítulo 10.

Con esto concluye la introducción al ajuste utilizando la API de `Trainer`. En el [Capítulo 7](/course/chapter7) se dará un ejemplo de cómo hacer esto para las tareas más comunes de PLN, pero ahora veamos cómo hacer lo mismo en PyTorch puro.

> [!TIP]
> ✏️ **¡Inténtalo!** Ajusta un modelo sobre el dataset GLUE SST-2 utilizando el procesamiento de datos que has implementado en la sección 2.


---



# Ajuste de un modelo con Keras


Una vez que hayas realizado todo el trabajo de preprocesamiento de datos de la última sección, sólo te quedan unos pocos pasos para entrenar el modelo. Sin embargo, ten en cuenta que el comando `model.fit()` se ejecutará muy lentamente en una CPU. Si no dispones de una GPU, puedes acceder a GPUs o TPUs gratuitas en [Google Colab](https://colab.research.google.com/).

Los siguientes ejemplos de código suponen que ya has ejecutado los ejemplos de la sección anterior. Aquí tienes un breve resumen de lo que necesitas:

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)
```

### Entrenamiento

Los modelos TensorFlow importados de 🤗 Transformers ya son modelos Keras. A continuación, una breve introducción a Keras.


**Video:** [Ver en YouTube](https://youtu.be/rnTGBy2ax1c)


Eso significa que, una vez que tenemos nuestros datos, se requiere muy poco trabajo para empezar a entrenar con ellos.


**Video:** [Ver en YouTube](https://youtu.be/AUozVp78dhk)


Como en el [capítulo anterior](/course/chapter2), utilizaremos la clase `TFAutoModelForSequenceClassification`, con dos etiquetas:

```py
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Observarás que, a diferencia del [Capítulo 2](/course/es/chapter2), aparece una advertencia después de instanciar este modelo preentrenado. Esto se debe a que BERT no ha sido preentrenado para la clasificación de pares de frases, por lo que la cabeza del modelo preentrenado se ha eliminado y en su lugar se ha añadido una nueva cabeza adecuada para la clasificación de secuencias. Las advertencias indican que algunos pesos no se han utilizado (los correspondientes a la cabeza de preentrenamiento eliminada) y que otros se han inicializado aleatoriamente (los correspondientes a la nueva cabeza). La advertencia concluye animándote a entrenar el modelo, que es exactamente lo que vamos a hacer ahora.

Para afinar el modelo en nuestro dataset, sólo tenemos que compilar nuestro modelo con `compile()` y luego pasar nuestros datos al método `fit()`. Esto iniciará el proceso de ajuste (que debería tardar un par de minutos en una GPU) e informará de la pérdida de entrenamiento a medida que avanza, además de la pérdida de validación al final de cada época.

> [!TIP]
> Ten en cuenta que los modelos 🤗 Transformers tienen una característica especial que la mayoría de los modelos Keras no tienen - pueden usar automáticamente una pérdida apropiada que calculan internamente. Usarán esta pérdida por defecto si no estableces un argumento de pérdida en `compile()`. Tea en cuenta que para utilizar la pérdida interna tendrás que pasar las etiquetas como parte de la entrada, en vez de como una etiqueta separada como es habitual en los modelos Keras. Veremos ejemplos de esto en la Parte 2 del curso, donde definir la función de pérdida correcta puede ser complicado. Para la clasificación de secuencias, sin embargo, una función de pérdida estándar de Keras funciona bien, así que eso es lo que usaremos aquí.

```py
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
)
```

> [!WARNING]
> Ten en cuenta un fallo muy común aquí: por poder, _puedes_ pasar simplemente el nombre de la función de pérdida como una cadena a Keras, pero por defecto Keras asumirá que ya has aplicado una función softmax a tus salidas. Sin embargo, muchos modelos devuelven los valores justo antes de que se aplique la función softmax, también conocidos como _logits_. Tenemos que decirle a la función de pérdida que eso es lo que hace nuestro modelo, y la única manera de hacerlo es llamándola directamente, en lugar de pasar su nombre con una cadena.

### Mejorar el rendimiento del entrenamiento


**Video:** [Ver en YouTube](https://youtu.be/cpzq6ESSM5c)


Si ejecutas el código anterior seguro que funciona, pero comprobarás que la pérdida sólo disminuye lenta o esporádicamente. La causa principal es la _tasa de aprendizaje_ (learning rate en inglés). Al igual que con la pérdida, cuando pasamos a Keras el nombre de un optimizador como una cadena, Keras inicializa ese optimizador con valores por defecto para todos los parámetros, incluyendo la tasa de aprendizaje. Sin embargo, por experiencia sabemos que los transformadores se benefician de una tasa de aprendizaje mucho menor que la predeterminada para Adam, que es 1e-3, también escrito
como 10 a la potencia de -3, o 0,001. 5e-5 (0,00005), que es unas veinte veces menor, es un punto de partida mucho mejor.

Además de reducir la tasa de aprendizaje, tenemos un segundo truco en la manga: podemos reducir lentamente la tasa de aprendizaje a lo largo del entrenamiento. En la literatura, a veces se habla de _decrecimiento_ o _reducción_ de la tasa de aprendizaje. En Keras, la mejor manera de hacer esto es utilizar un _programador de tasa de aprendizaje_. Una buena opción es `PolynomialDecay` que, a pesar del nombre, con la configuración por defecto simplemente hace que la tasa de aprendizaje decaiga decae linealmente desde el valor inicial hasta el valor final durante el transcurso del entrenamiento, que es exactamente lo que queremos. Con el fin de utilizar un programador correctamente, necesitamos decirle cuánto tiempo va a durar el entrenamiento. Especificamos esto a continuación como el número de pasos de entrenamiento: `num_train_steps`.

```py
from tensorflow.keras.optimizers.schedules import PolynomialDecay

batch_size = 8
num_epochs = 3

# El número de pasos de entrenamiento es el número de muestras del conjunto de datos
# dividido por el tamaño del lote y multiplicado por el número total de épocas.
# Ten en cuenta que el conjunto de datos tf_train_dataset es un conjunto de datos
# tf.data.Dataset por lotes, no el conjunto de datos original de Hugging Face
# por lo que su len() ya es num_samples // batch_size.

num_train_steps = len(tf_train_dataset) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=lr_scheduler)
```

> [!TIP]
> La librería 🤗 Transformers también tiene una función `create_optimizer()` que creará un optimizador `AdamW` con descenso de tasa de aprendizaje. Verás en detalle este útil atajo en próximas secciones del curso.

Ahora tenemos nuestro nuevo optimizador, y podemos intentar entrenar con él. En primer lugar, vamos a recargar el modelo, para restablecer los cambios en los pesos del entrenamiento que acabamos de hacer, y luego podemos compilarlo con el nuevo optimizador:

```py
import tensorflow as tf

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
```

Ahora, ajustamos de nuevo:

```py
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
```

> [!TIP]
> 💡 Si quieres subir automáticamente tu modelo a Hub durante el entrenamiento, puedes pasar un `PushToHubCallback` en el método `model.fit()`. Aprenderemos más sobre esto en el [Capítulo 4](/course/es/chapter4/3)

### Predicciones del Modelo[[model-predictions]]


**Video:** [Ver en YouTube](https://youtu.be/nx10eh4CoOs)


Entrenar y ver cómo disminuye la pérdida está muy bien, pero ¿qué pasa si queremos obtener la salida del modelo entrenado, ya sea para calcular algunas métricas o para utilizar el modelo en producción? Para ello, podemos utilizar el método `predict()`. Este devuelve los _logits_ de la cabeza de salida del modelo, uno por clase.

```py
preds = model.predict(tf_validation_dataset)["logits"]
```

Podemos convertir estos logits en predicciones de clases del modelo utilizando `argmax` para encontrar el logit más alto, que corresponde a la clase más probable:

```py
class_preds = np.argmax(preds, axis=1)
print(preds.shape, class_preds.shape)
```

```python out
(408, 2) (408,)
```

Ahora, ¡utilicemos esos `preds` (predicciones) para calcular métricas! Podemos cargar las métricas asociadas al conjunto de datos MRPC tan fácilmente como cargamos el conjunto de datos, esta vez con la función `evaluate.load()`. El objeto devuelto tiene un método `compute()` que podemos utilizar para calcular las métricas:

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=class_preds, references=raw_datasets["validation"]["label"])
```

```python out
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```

Los resultados exactos que obtengas pueden variar, ya que la inicialización aleatoria de la cabeza del modelo podría cambiar los valores resultantes de las métricas. Aquí, podemos ver que nuestro modelo tiene una precisión del 85,78% en el conjunto de validación y una puntuación F1 de 89,97. Estas son las dos métricas utilizadas para evaluar los resultados del conjunto de datos MRPC del benchmark GLUE. La tabla del [paper de BERT](https://arxiv.org/pdf/1810.04805.pdf) muestra una puntuación F1 de 88,9 para el modelo base. Se trataba del modelo `uncased` ("no encasillado"), mientras que nosotros utilizamos el modelo `cased` ("encasillado"), lo que explica el mejor resultado.

Con esto concluye la introducción al ajuste de modelos utilizando la API de Keras. En el [Capítulo 7](/course/es/chapter7) se dará un ejemplo de cómo hacer esto para las tareas de PLN más comunes. Si quieres perfeccionar tus habilidades con la API Keras, intenta ajustar un modelo con el conjunto de datos GLUE SST-2, utilizando el procesamiento de datos que hiciste en la sección 2.


---

# Un entrenamiento completo


**Video:** [Ver en YouTube](https://youtu.be/Dh9CL8fyG80)


Ahora veremos como obtener los mismos resultados de la última sección sin hacer uso de la clase `Trainer`. De nuevo, asumimos que has hecho el procesamiento de datos en la sección 2. Aquí mostramos un resumen que cubre todo lo que necesitarás.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Prepárate para el entrenamiento

Antes de escribir nuestro bucle de entrenamiento, necesitaremos definir algunos objetos. Los primeros son los `dataloaders` (literalmente, "cargadores de datos") que usaremos para iterar sobre lotes. Pero antes de que podamos definir esos `dataloaders`, necesitamos aplicar un poquito de preprocesamiento a nuestro `tokenized_datasets`, para encargarnos de algunas cosas que el `Trainer` hizo por nosotros de manera automática. Específicamente, necesitamos:

- Remover las columnas correspondientes a valores que el model no espera (como las columnas `sentence1` y `sentence2`).
- Renombrar la columna `label` con `labels` (porque el modelo espera el argumento llamado `labels`).
- Configurar el formato de los conjuntos de datos para que retornen tensores PyTorch en lugar de listas.

Nuestro `tokenized_datasets` tiene un método para cada uno de esos pasos:

```py
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

Ahora podemos verificar que el resultado solo tiene columnas que nuestro modelo aceptará:

```python
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

Ahora que esto esta hecho, es fácil definir nuestros `dataloaders`:

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

Para verificar rápidamente que no hubo errores en el procesamiento de datos, podemos inspeccionar un lote de la siguiente manera:

```py
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

```python out
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
```

Nótese que los tamaños serán un poco distintos en tu caso ya que configuramos `shuffle=True` para el dataloader de entrenamiento y estamos rellenando a la máxima longitud dentro del lote.

Ahora que hemos completado el preprocesamiento de datos (un objetivo gratificante y al mismo tiempo elusivo para cual cualquier practicante de ML), enfoquémonos en el modelo. Lo vamos a crear exactamente como lo hicimos en la sección anterior.

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Para asegurarnos de que todo va a salir sin problems durante el entrenamiento, vamos a pasar un lote a este modelo:

```py
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

```python out
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

Todos los modelos 🤗 Transformers van a retornar la pérdida cuando se pasan los `labels`, y también obtenemos los logits (dos por cada entrada en nuestro lote, asi que es un tensor de tamaño 8 x 2).

Estamos casi listos para escribir nuestro bucle de entrenamiento! Nos están faltando dos cosas: un optimizador y un programador de la tasa de aprendizaje. Ya que estamos tratando de replicar a mano lo que el `Trainer` estaba haciendo, usaremos los mismos valores por defecto. El optimizador usado por el `Trainer` es `AdamW`, que es el mismo que Adam, pero con un cambio para la regularización de decremento de los pesos (ver ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) por Ilya Loshchilov y Frank Hutter):

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

Finalmente, el programador por defecto de la tasa de aprendizaje es un decremento lineal desde al valor máximo (5e-5) hasta 0. Para definirlo apropiadamente, necesitamos saber el número de pasos de entrenamiento que vamos a tener, el cual viene dado por el número de épocas que deseamos correr multiplicado por el número de lotes de entrenamiento (que es el largo de nuestro dataloader de entrenamiento). El `Trainer` usa tres épocas por defecto, asi que usaremos eso:

```py
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```

```python out
1377
```

### El bucle de entrenamiento

Una última cosa: vamos a querer usar el GPU si tenemos acceso a uno (en un CPU, el entrenamiento puede tomar varias horas en lugar de unos pocos minutos). Para hacer esto, definimos un `device` sobre el que pondremos nuestro modelo y nuestros lotes:

```py
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```

```python out
device(type='cuda')
```

¡Ya está todo listo para entrenar! Para tener una idea de cuándo va a terminar el entrenamiento, adicionamos una barra de progreso sobre el número de pasos de entrenamiento, usando la librería `tqdm`:

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Puedes ver que la parte central del bucle de entrenamiento luce bastante como el de la introducción. No se incluyó ningún tipo de reportes, asi que este bucle de entrenamiento no va a indicar como se esta desempeñando el modelo. Para eso necesitamos añadir un bucle de evaluación.

### El bucle de evaluación

Como lo hicimos anteriormente, usaremos una métrica ofrecida por la librería 🤗 Evaluate. Ya hemos visto el método `metric.compute()`, pero de hecho las métricas se pueden acumular sobre los lotes a medida que avanzamos en el bucle de predicción con el método `add_batch()`. Una vez que hemos acumulado todos los lotes, podemos obtener el resultado final con `metric.compute()`. Aquí se muestra cómo se puede implementar en un bucle de evaluación:

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

```python out
{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```

De nuevo, tus resultados serán un tanto diferente debido a la inicialización aleatoria en la cabeza del modelo y el mezclado de los datos, pero deberían tener valores similares.

> [!TIP]
> ✏️ **Inténtalo!** Modifica el bucle de entrenamiento anterior para ajustar tu modelo en el conjunto de datos SST-2.

### Repotencia tu bucle de entrenamiento con Accelerate 🤗


**Video:** [Ver en YouTube](https://youtu.be/s7dy8QRgjJ0)


El bucle de entrenamiento que definimos anteriormente trabaja bien en una sola CPU o GPU. Pero usando la librería [Accelerate 🤗](https://github.com/huggingface/accelerate), con solo pocos ajustes podemos habilitar el entrenamiento distribuido en múltiples GPUs o CPUs. Comenzando con la creación de los dataloaders de entrenamiento y validación, aquí se muestra como luce nuestro bucle de entrenamiento:

```py
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Y aquí están los cambios:

```diff
+ from accelerate import Accelerator
  from torch.optim import AdamW
  from transformers import AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

La primera línea a agregarse es la línea del `import`. La segunda línea crea un objeto `Accelerator` que revisa el ambiente e inicializa la configuración distribuida apropiada. La librería 🤗 Accelerate se encarga de asignarte el dispositivo, para que puedas remover las líneas que ponen el modelo en el dispositivo (o si prefieres, cámbialas para usar el `accelerator.device` en lugar de `device`).

Ahora la mayor parte del trabajo se hace en la línea que envía los `dataloaders`, el modelo y el optimizador al `accelerator.prepare()`. Este va a envolver esos objetos en el contenedor apropiado para asegurarse que tu entrenamiento distribuido funcione como se espera. Los cambios que quedan son remover la línea que coloca el lote en el `device` (de nuevo, si deseas dejarlo así bastaría con cambiarlo para que use el `accelerator.device`) y reemplazar `loss.backward()` con `accelerator.backward(loss)`.

> [!TIP]
> ⚠️ Para obtener el beneficio de la aceleración ofrecida por los TPUs de la
>   nube, recomendamos rellenar las muestras hasta una longitud fija con los
>   argumentos `padding="max_length"` y `max_length` del tokenizador.

Si deseas copiarlo y pegarlo para probar, así es como luce el bucle completo de entrenamiento con 🤗 Accelerate:

```py
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Colocando esto en un script `train.py` permitirá que el mismo sea ejecutable en cualquier configuración distribuida. Para probarlo en tu configuración distribuida, ejecuta el siguiente comando:

```bash
accelerate config
```

el cual hará algunas preguntas y guardará tus respuestas en un archivo de configuración usado por este comando:

```
accelerate launch train.py
```

el cual iniciará en entrenamiento distribuido.

Si deseas ejecutar esto en un Notebook (por ejemplo, para probarlo con TPUs en Colab), solo pega el código en una `training_function()` y ejecuta la última celda con:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

Puedes encontrar más ejemplos en el [repositorio 🤗 Accelerate](https://github.com/huggingface/accelerate/tree/main/examples).


---



# Ajuste de modelos, ¡hecho!


¡Qué divertido! En los dos primeros capítulos aprendiste sobre modelos y tokenizadores, y ahora sabes cómo ajustarlos a tus propios datos. Para recapitular, en este capítulo:


**PyTorch:**

- Aprendiste sobre los conjuntos de datos del [Hub](https://huggingface.co/datasets)
- Aprendiste a cargar y preprocesar conjuntos de datos, incluyendo el uso de padding dinámico y los "collators"
- Implementaste tu propio ajuste (fine-tuning) y cómo evaluar un modelo
- Implementaste un bucle de entrenamiento de bajo nivel
- Utilizaste 🤗 Accelerate para adaptar fácilmente tu bucle de entrenamiento para que funcione en múltiples GPUs o TPUs

**TensorFlow/Keras:**

- Aprendiste sobre los conjuntos de datos en [Hub](https://huggingface.co/datasets)
- Aprendiste a cargar y preprocesar conjuntos de datos
- Aprendiste a ajustar (fine-tuning) y evaluar un modelo con Keras
- Implementaste una métrica personalizada



---




# Quiz de final de capítulo


A ver qué has aprendido en este capítulo:

### 1. El dataset `emotion` contiene mensajes de Twitter etiquetados con emociones. Búscalo en el [Hub](https://huggingface.co/datasets), y lee la tarjeta del dataset. ¿Cuál de estas no es una de sus emociones básicas?


- Alegría
- Amor
- Confusión
- Sorpresa


### 2. Busca el dataset `ar_sarcasm` en el [Hub](https://huggingface.co/datasets). ¿Con qué tarea es compatible?


- Clasificación de sentimientos
- Traducción automática
- Reconocimiento de entidades nombradas
- Responder preguntas


### 3. ¿Cómo se procesan un par de frases según el modelo BERT?


- tokens_frase_1 [SEP] tokens_frase_2
- [CLS] tokens_frase_1 tokens_frase_2
- [CLS] tokens_frase_1 [SEP] tokens_frase_2 [SEP]
- [CLS] tokens_frase_1 [SEP] tokens_frase_2


**PyTorch:**

### 4. ¿Cuáles son las ventajas del método `Dataset.map()`?


- Los resultados de la función se almacenan en caché, por lo que no tardaremos nada en volver a ejecutar el código.
- Puede aplicar multiprocesamiento para ir más rápido que si se aplicara la función a cada elemento del conjunto de datos.
- No carga todo el conjunto de datos en memoria, sino que guarda los resultados en cuanto se procesa un elemento.


### 5. ¿Qué significa padding dinámico?


- Es cuando se rellenan las entradas de cada lote con la longitud máxima de todo el conjunto de datos.
- Es cuando rellenas tus entradas cuando se crea el lote, a la longitud máxima de las frases de ese lote.
- Es cuando se rellenan las entradas para que cada frase tenga el mismo número de tokens que la anterior en el conjunto de datos.


### 6. ¿Cuál es el objetivo de la función "collate"?


- Se asegura de que todas las secuencias del conjunto de datos tengan la misma longitud.
- Combina todas las muestras del conjunto de datos en un lote.
- Preprocesa todo el conjunto de datos.
- Trunca las secuencias del conjunto de datos.


### 7. ¿Qué ocurre cuando instancias una de las clases `AutoModelForXxx` con un modelo del lenguaje preentrenado (como `bert-base-uncased`) que corresponde a una tarea distinta de aquella para la que fue entrenado?


- Nada, pero recibes una advertencia.
- La cabeza del modelo preentrenado se elimina y en su lugar se inserta una nueva cabeza adecuada para la tarea.
- La cabeza del modelo preentrenado es eliminada.
- Nada, ya que el modelo se puede seguir ajustando para la otra tarea.


### 8. ¿Para qué sirve `TrainingArguments`?


- Contiene todos los hiperparámetros utilizados para el entrenamiento y la evaluación con `Trainer`.
- Especifica el tamaño del modelo.
- Solo contiene los hiperparámetros utilizados para la evaluación.
- Solo contiene los hiperparámetros utilizados para el entrenamiento.


### 9. ¿Por qué deberías utilizar la librería 🤗 Accelerate?


- Facilita acceso a modelos más rápidos.
- Proporciona una API de alto nivel para que no tenga que implementar mi propio bucle de entrenamiento.
- Hace que nuestros bucles de entrenamiento funcionen con estrategias distribuidas.
- Ofrece más funciones de optimización.


**TensorFlow/Keras:**

### 4. ¿Qué ocurre cuando instancias una de las clases `TFAutoModelForXxx` con un modelo del lenguaje preentrenado (como `bert-base-uncased`) que corresponde a una tarea distinta de aquella para la que fue entrenado?


- Nada, pero recibes una advertencia.
- La cabeza del modelo preentrenado se elimina y en su lugar se inserta una nueva cabeza adecuada para la tarea.
- La cabeza del modelo preentrenado es eliminada
- Nada, ya que el modelo se puede seguir ajustando para la otra tarea.


### 5. Los modelos TensorFlow de `transformers` ya son modelos Keras. ¿Qué ventajas ofrece esto?


- Los modelos funcionan directamente en una TPU.
- Puede aprovechar los métodos existentes, como `compile()`, `fit()` y `predict()`.
- Tienes la oportunidad de aprender Keras a la vez que transformadores.
- Puede calcular fácilmente las métricas relacionadas con el dataset.


### 6. ¿Cómo puedes definir tu propia métrica personalizada?


- Creando una subclase de `tf.keras.metrics.Metric`.
- Utilizando la API funcional de Keras.
- Utilizando una función cuya firma sea `metric_fn(y_true, y_pred)`.
- Buscándolo en Google.




---

# 4. Compartiendo modelos y tokenizadores

# El Hub de Hugging Face[[the-hugging-face-hub]]


El [Hub de Hugging Face](https://huggingface.co/) –- nuestro sitio web principal –- es una plataforma central que permite a cualquier persona descubrir, usar y contribuir con nuevos modelos y conjuntos de datos de ultima generacion. Aloja una amplia variedad de modelos, con mas de 10,000 disponibles publicamente. Nos centraremos en los modelos en este capitulo, y examinaremos los conjuntos de datos en el Capitulo 5.

Los modelos en el Hub no estan limitados a 🤗 Transformers o incluso a NLP. Hay modelos de [Flair](https://github.com/flairNLP/flair) y [AllenNLP](https://github.com/allenai/allennlp) para NLP, [Asteroid](https://github.com/asteroid-team/asteroid) y [pyannote](https://github.com/pyannote/pyannote-audio) para audio, y [timm](https://github.com/rwightman/pytorch-image-models) para vision, por nombrar algunos.

Cada uno de estos modelos esta alojado como un repositorio Git, lo que permite versionado y reproducibilidad. Compartir un modelo en el Hub significa abrirlo a la comunidad y hacerlo accesible a cualquier persona que busque usarlo facilmente, a su vez eliminando su necesidad de entrenar un modelo por su cuenta y simplificando el intercambio y uso.

Adicionalmente, compartir un modelo en el Hub automaticamente despliega una API de Inferencia alojada para ese modelo. Cualquier persona en la comunidad es libre de probarlo directamente en la pagina del modelo, con entradas personalizadas y widgets apropiados.

La mejor parte es que compartir y usar cualquier modelo publico en el Hub es completamente gratis! Tambien existen [planes pagos](https://huggingface.co/pricing) si deseas compartir modelos de forma privada.

El video a continuacion muestra como navegar por el Hub.


**Video:** [Ver en YouTube](https://youtu.be/XvSGPZFEjDY)


Se requiere tener una cuenta de huggingface.co para seguir esta parte, ya que crearemos y administraremos repositorios en el Hub de Hugging Face: [crear una cuenta](https://huggingface.co/join)


---



# Usando modelos preentrenados[[using-pretrained-models]]


El Model Hub hace que seleccionar el modelo apropiado sea simple, de modo que usarlo en cualquier biblioteca posterior se pueda hacer en unas pocas lineas de codigo. Veamos como usar realmente uno de estos modelos, y como contribuir de vuelta a la comunidad.

Digamos que estamos buscando un modelo basado en frances que pueda realizar relleno de mascara.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/camembert.gif" alt="Seleccionando el modelo Camembert." width="80%"/>
</div>

Seleccionamos el checkpoint `camembert-base` para probarlo. El identificador `camembert-base` es todo lo que necesitamos para comenzar a usarlo! Como has visto en capitulos anteriores, podemos instanciarlo usando la funcion `pipeline()`:

```py
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```

```python out
[
  {'sequence': 'Le camembert est délicieux :)', 'score': 0.49091005325317383, 'token': 7200, 'token_str': 'délicieux'},
  {'sequence': 'Le camembert est excellent :)', 'score': 0.1055697426199913, 'token': 2183, 'token_str': 'excellent'},
  {'sequence': 'Le camembert est succulent :)', 'score': 0.03453313186764717, 'token': 26202, 'token_str': 'succulent'},
  {'sequence': 'Le camembert est meilleur :)', 'score': 0.0330314114689827, 'token': 528, 'token_str': 'meilleur'},
  {'sequence': 'Le camembert est parfait :)', 'score': 0.03007650189101696, 'token': 1654, 'token_str': 'parfait'}
]
```

Como puedes ver, cargar un modelo dentro de un pipeline es extremadamente simple. Lo unico que debes tener en cuenta es que el checkpoint elegido sea adecuado para la tarea en la que se va a usar. Por ejemplo, aqui estamos cargando el checkpoint `camembert-base` en el pipeline `fill-mask`, lo cual es completamente correcto. Pero si cargaramos este checkpoint en el pipeline `text-classification`, los resultados no tendrian ningun sentido porque la cabeza de `camembert-base` no es adecuada para esta tarea! Recomendamos usar el selector de tareas en la interfaz del Hub de Hugging Face para seleccionar los checkpoints apropiados:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/tasks.png" alt="El selector de tareas en la interfaz web." width="80%"/>
</div>

Tambien puedes instanciar el checkpoint usando la arquitectura del modelo directamente:


**PyTorch:**

```py
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```

Sin embargo, recomendamos usar las [clases `Auto*`](https://huggingface.co/transformers/model_doc/auto?highlight=auto#auto-classes) en su lugar, ya que estan disenadas para ser agnosticas de la arquitectura. Mientras el ejemplo de codigo anterior limita a los usuarios a checkpoints cargables en la arquitectura CamemBERT, usar las clases `Auto*` hace que cambiar checkpoints sea simple:

```py
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

**TensorFlow/Keras:**

```py
from transformers import CamembertTokenizer, TFCamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = TFCamembertForMaskedLM.from_pretrained("camembert-base")
```

Sin embargo, recomendamos usar las [clases `TFAuto*`](https://huggingface.co/transformers/model_doc/auto?highlight=auto#auto-classes) en su lugar, ya que estan disenadas para ser agnosticas de la arquitectura. Mientras el ejemplo de codigo anterior limita a los usuarios a checkpoints cargables en la arquitectura CamemBERT, usar las clases `TFAuto*` hace que cambiar checkpoints sea simple:

```py
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = TFAutoModelForMaskedLM.from_pretrained("camembert-base")
```


> [!TIP]
> Cuando uses un modelo preentrenado, asegurate de verificar como fue entrenado, en que conjuntos de datos, sus limites y sus sesgos. Toda esta informacion deberia estar indicada en su tarjeta de modelo.


---



# Compartiendo modelos preentrenados[[sharing-pretrained-models]]


En los pasos a continuacion, examinaremos las formas mas faciles de compartir modelos preentrenados en el 🤗 Hub. Hay herramientas y utilidades disponibles que hacen simple compartir y actualizar modelos directamente en el Hub, las cuales exploraremos a continuacion.


**Video:** [Ver en YouTube](https://youtu.be/9yY3RB_GSPM)


Animamos a todos los usuarios que entrenan modelos a contribuir compartiendolos con la comunidad — compartir modelos, incluso cuando son entrenados en conjuntos de datos muy especificos, ayudara a otros, ahorrandoles tiempo y recursos computacionales y proporcionando acceso a artefactos entrenados utiles. A su vez, tu puedes beneficiarte del trabajo que otros han hecho!

Hay tres formas de crear nuevos repositorios de modelos:

- Usando la API `push_to_hub`
- Usando la biblioteca Python `huggingface_hub`
- Usando la interfaz web

Una vez que hayas creado un repositorio, puedes subir archivos a el via git y git-lfs. Te guiaremos a traves de la creacion de repositorios de modelos y la subida de archivos a ellos en las siguientes secciones.

## Usando la API `push_to_hub`[[using-the-pushtohub-api]]


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/Zh0FfmVrKX0)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/pUh5cGmNV8Y)


La forma mas simple de subir archivos al Hub es aprovechando la API `push_to_hub`.

Antes de continuar, necesitaras generar un token de autenticacion para que la API `huggingface_hub` sepa quien eres y a que espacios de nombres tienes acceso de escritura. Asegurate de estar en un entorno donde tengas `transformers` instalado (ver [Configuracion](/course/chapter0)). Si estas en un notebook, puedes usar la siguiente funcion para iniciar sesion:

```python
from huggingface_hub import notebook_login

notebook_login()
```

En una terminal, puedes ejecutar:

```bash
huggingface-cli login
```

En ambos casos, se te pedira tu nombre de usuario y contrasena, que son los mismos que usas para iniciar sesion en el Hub. Si aun no tienes un perfil en el Hub, deberias crear uno [aqui](https://huggingface.co/join).

Genial! Ahora tienes tu token de autenticacion almacenado en tu carpeta de cache. Creemos algunos repositorios!


**PyTorch:**

Si has jugado con la API `Trainer` para entrenar un modelo, la forma mas facil de subirlo al Hub es establecer `push_to_hub=True` cuando definas tus `TrainingArguments`:

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
```

Cuando llames a `trainer.train()`, el `Trainer` entonces subira tu modelo al Hub cada vez que sea guardado (aqui cada epoca) en un repositorio en tu espacio de nombres. Ese repositorio tendra el nombre del directorio de salida que elegiste (aqui `bert-finetuned-mrpc`) pero puedes elegir un nombre diferente con `hub_model_id = "un_nombre_diferente"`.

Para subir tu modelo a una organizacion de la que eres miembro, simplemente pasala con `hub_model_id = "mi_organizacion/nombre_de_mi_repo"`.

Una vez que tu entrenamiento haya terminado, deberias hacer un `trainer.push_to_hub()` final para subir la ultima version de tu modelo. Tambien generara una tarjeta de modelo con todos los metadatos relevantes, reportando los hiperparametros usados y los resultados de evaluacion! Aqui hay un ejemplo del contenido que podrias encontrar en una tarjeta de modelo asi:

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/model_card.png" alt="Un ejemplo de una tarjeta de modelo autogenerada." width="100%"/>
</div>

**TensorFlow/Keras:**

Si estas usando Keras para entrenar tu modelo, la forma mas facil de subirlo al Hub es pasar un `PushToHubCallback` cuando llames a `model.fit()`:

```py
from transformers import PushToHubCallback

callback = PushToHubCallback(
    "bert-finetuned-mrpc", save_strategy="epoch", tokenizer=tokenizer
)
```

Luego deberias agregar `callbacks=[callback]` en tu llamada a `model.fit()`. El callback entonces subira tu modelo al Hub cada vez que sea guardado (aqui cada epoca) en un repositorio en tu espacio de nombres. Ese repositorio tendra el nombre del directorio de salida que elegiste (aqui `bert-finetuned-mrpc`) pero puedes elegir un nombre diferente con `hub_model_id = "un_nombre_diferente"`.

Para subir tu modelo a una organizacion de la que eres miembro, simplemente pasala con `hub_model_id = "mi_organizacion/nombre_de_mi_repo"`.


A un nivel mas bajo, acceder al Model Hub puede hacerse directamente en modelos, tokenizadores y objetos de configuracion via su metodo `push_to_hub()`. Este metodo se encarga tanto de la creacion del repositorio como de subir los archivos del modelo y tokenizador directamente al repositorio. No se requiere manejo manual, a diferencia de la API que veremos a continuacion.

Para tener una idea de como funciona, primero inicialicemos un modelo y un tokenizador:


**PyTorch:**

```py
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

**TensorFlow/Keras:**

```py
from transformers import TFAutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```


Eres libre de hacer lo que quieras con estos — agregar tokens al tokenizador, entrenar el modelo, ajustarlo. Una vez que estes satisfecho con el modelo, pesos y tokenizador resultantes, puedes aprovechar el metodo `push_to_hub()` disponible directamente en el objeto `model`:

```py
model.push_to_hub("dummy-model")
```

Esto creara el nuevo repositorio `dummy-model` en tu perfil, y lo poblara con tus archivos de modelo. Haz lo mismo con el tokenizador, para que todos los archivos esten ahora disponibles en este repositorio:

```py
tokenizer.push_to_hub("dummy-model")
```

Si perteneces a una organizacion, simplemente especifica el argumento `organization` para subir al espacio de nombres de esa organizacion:

```py
tokenizer.push_to_hub("dummy-model", organization="huggingface")
```

Si deseas usar un token de Hugging Face especifico, eres libre de especificarlo al metodo `push_to_hub()` tambien:

```py
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")
```

Ahora ve al Model Hub para encontrar tu modelo recien subido: *https://huggingface.co/user-or-organization/dummy-model*.

Haz clic en la pestana "Files and versions", y deberias ver los archivos visibles en la siguiente captura de pantalla:


**PyTorch:**

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/push_to_hub_dummy_model.png" alt="Modelo dummy que contiene tanto los archivos del tokenizador como del modelo." width="80%"/>
</div>

**TensorFlow/Keras:**

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/push_to_hub_dummy_model_tf.png" alt="Modelo dummy que contiene tanto los archivos del tokenizador como del modelo." width="80%"/>
</div>


> [!TIP]
> ✏️ **Pruebalo!** Toma el modelo y tokenizador asociados con el checkpoint `bert-base-cased` y subeloss a un repo en tu espacio de nombres usando el metodo `push_to_hub()`. Verifica que el repo aparezca correctamente en tu pagina antes de eliminarlo.

Como has visto, el metodo `push_to_hub()` acepta varios argumentos, lo que hace posible subir a un repositorio especifico o espacio de nombres de organizacion, o usar un token de API diferente. Te recomendamos echar un vistazo a la especificacion del metodo disponible directamente en la [documentacion de 🤗 Transformers](https://huggingface.co/transformers/model_sharing) para tener una idea de lo que es posible.

El metodo `push_to_hub()` esta respaldado por el paquete Python [`huggingface_hub`](https://github.com/huggingface/huggingface_hub), que ofrece una API directa al Hub de Hugging Face. Esta integrado dentro de 🤗 Transformers y varias otras bibliotecas de machine learning, como [`allenlp`](https://github.com/allenai/allennlp). Aunque nos enfocamos en la integracion con 🤗 Transformers en este capitulo, integrarlo en tu propio codigo o biblioteca es simple.

Salta a la ultima seccion para ver como subir archivos a tu repositorio recien creado!

## Usando la biblioteca Python `huggingface_hub`[[using-the-huggingfacehub-python-library]]

La biblioteca Python `huggingface_hub` es un paquete que ofrece un conjunto de herramientas para los hubs de modelos y conjuntos de datos. Proporciona metodos y clases simples para tareas comunes como obtener informacion sobre repositorios en el hub y administrarlos. Proporciona APIs simples que funcionan sobre git para administrar el contenido de esos repositorios e integrar el Hub en tus proyectos y bibliotecas.

Similar a usar la API `push_to_hub`, esto requerira que tengas tu token de API guardado en tu cache. Para hacer esto, necesitaras usar el comando `login` de la CLI, como se menciono en la seccion anterior (de nuevo, asegurate de anteponer estos comandos con el caracter `!` si estas ejecutando en Google Colab):

```bash
huggingface-cli login
```

El paquete `huggingface_hub` ofrece varios metodos y clases que son utiles para nuestro proposito. Primero, hay algunos metodos para administrar la creacion, eliminacion y otros de repositorios:

```python no-format
from huggingface_hub import (
    # Administracion de usuarios
    login,
    logout,
    whoami,

    # Creacion y administracion de repositorios
    create_repo,
    delete_repo,
    update_repo_visibility,

    # Y algunos metodos para recuperar/cambiar informacion sobre el contenido
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
```

Adicionalmente, ofrece la muy poderosa clase `Repository` para administrar un repositorio local. Exploraremos estos metodos y esa clase en las proximas secciones para entender como aprovecharlos.

El metodo `create_repo` puede usarse para crear un nuevo repositorio en el hub:

```py
from huggingface_hub import create_repo

create_repo("dummy-model")
```

Esto creara el repositorio `dummy-model` en tu espacio de nombres. Si lo deseas, puedes especificar a que organizacion deberia pertenecer el repositorio usando el argumento `organization`:

```py
from huggingface_hub import create_repo

create_repo("dummy-model", organization="huggingface")
```

Esto creara el repositorio `dummy-model` en el espacio de nombres `huggingface`, asumiendo que perteneces a esa organizacion. Otros argumentos que pueden ser utiles son:

- `private`, para especificar si el repositorio deberia ser visible para otros o no.
- `token`, si deseas anular el token almacenado en tu cache por un token dado.
- `repo_type`, si deseas crear un `dataset` o un `space` en lugar de un modelo. Los valores aceptados son `"dataset"` y `"space"`.

Una vez que el repositorio este creado, debemos agregarle archivos! Salta a la siguiente seccion para ver las tres formas en que esto puede manejarse.

## Usando la interfaz web[[using-the-web-interface]]

La interfaz web ofrece herramientas para administrar repositorios directamente en el Hub. Usando la interfaz, puedes facilmente crear repositorios, agregar archivos (incluso grandes!), explorar modelos, visualizar diffs y mucho mas.

Para crear un nuevo repositorio, visita [huggingface.co/new](https://huggingface.co/new):

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/new_model.png" alt="Pagina que muestra el modelo usado para la creacion de un nuevo repositorio de modelo." width="80%"/>
</div>

Primero, especifica el propietario del repositorio: este puede ser tu o cualquiera de las organizaciones a las que estas afiliado. Si eliges una organizacion, el modelo sera destacado en la pagina de la organizacion y cada miembro de la organizacion tendra la capacidad de contribuir al repositorio.

Luego, ingresa el nombre de tu modelo. Este tambien sera el nombre del repositorio. Finalmente, puedes especificar si quieres que tu modelo sea publico o privado. Los modelos privados estan ocultos de la vista publica.

Despues de crear tu repositorio de modelo, deberias ver una pagina como esta:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/empty_model.png" alt="Una pagina de modelo vacia despues de crear un nuevo repositorio." width="80%"/>
</div>

Aqui es donde tu modelo sera alojado. Para comenzar a poblarlo, puedes agregar un archivo README directamente desde la interfaz web.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/dummy_model.png" alt="El archivo README mostrando las capacidades de Markdown." width="80%"/>
</div>

El archivo README esta en Markdown — sientete libre de experimentar con el! La tercera parte de este capitulo esta dedicada a construir una tarjeta de modelo. Estas son de primera importancia para dar valor a tu modelo, ya que es donde le dices a otros lo que puede hacer.

Si miras la pestana "Files and versions", veras que no hay muchos archivos ahi todavia — solo el *README.md* que acabas de crear y el archivo *.gitattributes* que mantiene registro de archivos grandes.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/files.png" alt="La pestana 'Files and versions' solo muestra los archivos .gitattributes y README.md." width="80%"/>
</div>

Veremos como agregar algunos archivos nuevos a continuacion.

## Subiendo los archivos del modelo[[uploading-the-model-files]]

El sistema para administrar archivos en el Hub de Hugging Face esta basado en git para archivos regulares, y git-lfs (que significa [Git Large File Storage](https://git-lfs.github.com/)) para archivos mas grandes.

En la siguiente seccion, repasamos tres formas diferentes de subir archivos al Hub: a traves de `huggingface_hub` y a traves de comandos git.

### El enfoque `upload_file`[[the-uploadfile-approach]]

Usar `upload_file` no requiere que git y git-lfs esten instalados en tu sistema. Sube archivos directamente al 🤗 Hub usando solicitudes HTTP POST. Una limitacion de este enfoque es que no maneja archivos que sean mas grandes de 5GB de tamano. Si tus archivos son mas grandes de 5GB, por favor sigue los otros dos metodos detallados a continuacion.

La API puede usarse de la siguiente manera:

```py
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)
```

Esto subira el archivo `config.json` disponible en `<path_to_file>` a la raiz del repositorio como `config.json`, al repositorio `dummy-model`. Otros argumentos que pueden ser utiles son:

- `token`, si deseas anular el token almacenado en tu cache por un token dado.
- `repo_type`, si deseas subir a un `dataset` o un `space` en lugar de un modelo. Los valores aceptados son `"dataset"` y `"space"`.

### La clase `Repository`[[the-repository-class]]

La clase `Repository` administra un repositorio local de manera similar a git. Abstrae la mayoria de los puntos problematicos que uno puede tener con git para proporcionar todas las caracteristicas que requerimos.

Usar esta clase requiere tener git y git-lfs instalados, asi que asegurate de tener git-lfs instalado (ver [aqui](https://git-lfs.github.com/) para instrucciones de instalacion) y configurado antes de comenzar.

Para comenzar a jugar con el repositorio que acabamos de crear, podemos empezar inicializandolo en una carpeta local clonando el repositorio remoto:

```py
from huggingface_hub import Repository

repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")
```

Esto creo la carpeta `<path_to_dummy_folder>` en nuestro directorio de trabajo. Esta carpeta solo contiene el archivo `.gitattributes` ya que ese es el unico archivo creado al instanciar el repositorio a traves de `create_repo`.

Desde este punto, podemos aprovechar varios de los metodos tradicionales de git:

```py
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
```

Y otros! Recomendamos echar un vistazo a la documentacion de `Repository` disponible [aqui](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub#advanced-programmatic-repository-management) para una vision general de todos los metodos disponibles.

Actualmente, tenemos un modelo y un tokenizador que nos gustaria subir al hub. Hemos clonado exitosamente el repositorio, por lo tanto podemos guardar los archivos dentro de ese repositorio.

Primero nos aseguramos de que nuestro clon local este actualizado obteniendo los ultimos cambios:

```py
repo.git_pull()
```

Una vez hecho eso, guardamos los archivos del modelo y tokenizador:

```py
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

El `<path_to_dummy_folder>` ahora contiene todos los archivos del modelo y tokenizador. Seguimos el flujo de trabajo usual de git agregando archivos al area de preparacion, haciendo commit y subiendo al hub:

```py
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
```

Felicidades! Acabas de subir tus primeros archivos al hub.

### El enfoque basado en git[[the-git-based-approach]]

Este es el enfoque mas basico para subir archivos: lo haremos con git y git-lfs directamente. La mayor parte de la dificultad es abstraida por los enfoques anteriores, pero hay algunas advertencias con el siguiente metodo asi que seguiremos un caso de uso mas complejo.

Usar esta clase requiere tener git y git-lfs instalados, asi que asegurate de tener [git-lfs](https://git-lfs.github.com/) instalado (ver aqui para instrucciones de instalacion) y configurado antes de comenzar.

Primero comienza inicializando git-lfs:

```bash
git lfs install
```

```bash
Updated git hooks.
Git LFS initialized.
```

Una vez hecho eso, el primer paso es clonar tu repositorio de modelo:

```bash
git clone https://huggingface.co/<namespace>/<your-model-id>
```

Mi nombre de usuario es `lysandre` y he usado el nombre de modelo `dummy`, asi que para mi el comando termina luciendo asi:

```
git clone https://huggingface.co/lysandre/dummy
```

Ahora tengo una carpeta llamada *dummy* en mi directorio de trabajo. Puedo entrar a la carpeta con `cd` y echar un vistazo a los contenidos:

```bash
cd dummy && ls
```

```bash
README.md
```

Si acabas de crear tu repositorio usando el metodo `create_repo` del Hub de Hugging Face, esta carpeta solo deberia contener un archivo oculto `.gitattributes`. Si seguiste las instrucciones en la seccion anterior para crear un repositorio usando la interfaz web, la carpeta deberia contener un solo archivo *README.md* junto al archivo oculto `.gitattributes`, como se muestra aqui.

Agregar un archivo de tamano regular, como un archivo de configuracion, un archivo de vocabulario, o basicamente cualquier archivo menor a unos pocos megabytes, se hace exactamente como se haria en cualquier sistema basado en git. Sin embargo, archivos mas grandes deben registrarse a traves de git-lfs para poder subirlos a *huggingface.co*.

Volvamos a Python por un momento para generar un modelo y tokenizador que nos gustaria enviar a nuestro repositorio dummy:


**PyTorch:**

```py
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Haz lo que quieras con el modelo, entrenalo, ajustalo...

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

**TensorFlow/Keras:**

```py
from transformers import TFAutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Haz lo que quieras con el modelo, entrenalo, ajustalo...

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```


Ahora que hemos guardado algunos artefactos de modelo y tokenizador, echemos otro vistazo a la carpeta *dummy*:

```bash
ls
```


**PyTorch:**

```bash
config.json  pytorch_model.bin  README.md  sentencepiece.bpe.model  special_tokens_map.json tokenizer_config.json  tokenizer.json
```

Si miras los tamanos de archivo (por ejemplo, con `ls -lh`), deberias ver que el archivo de diccionario de estado del modelo (*pytorch_model.bin*) es el unico atipico, con mas de 400 MB.

**TensorFlow/Keras:**

```bash
config.json  README.md  sentencepiece.bpe.model  special_tokens_map.json  tf_model.h5  tokenizer_config.json  tokenizer.json
```

Si miras los tamanos de archivo (por ejemplo, con `ls -lh`), deberias ver que el archivo de diccionario de estado del modelo (*t5_model.h5*) es el unico atipico, con mas de 400 MB.


> [!TIP]
> ✏️ Al crear el repositorio desde la interfaz web, el archivo *.gitattributes* se configura automaticamente para considerar archivos con ciertas extensiones, como *.bin* y *.h5*, como archivos grandes, y git-lfs los rastreara sin ninguna configuracion necesaria de tu parte.

Ahora podemos continuar y proceder como normalmente lo hariamos con repositorios Git tradicionales. Podemos agregar todos los archivos al entorno de preparacion de Git usando el comando `git add`:

```bash
git add .
```

Luego podemos echar un vistazo a los archivos que estan actualmente preparados:

```bash
git status
```


**PyTorch:**

```bash
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
  modified:   .gitattributes
	new file:   config.json
	new file:   pytorch_model.bin
	new file:   sentencepiece.bpe.model
	new file:   special_tokens_map.json
	new file:   tokenizer.json
	new file:   tokenizer_config.json
```

**TensorFlow/Keras:**

```bash
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
  modified:   .gitattributes
  	new file:   config.json
	new file:   sentencepiece.bpe.model
	new file:   special_tokens_map.json
	new file:   tf_model.h5
	new file:   tokenizer.json
	new file:   tokenizer_config.json
```


Similarmente, podemos asegurarnos de que git-lfs este rastreando los archivos correctos usando su comando `status`:

```bash
git lfs status
```


**PyTorch:**

```bash
On branch main
Objects to be pushed to origin/main:


Objects to be committed:

	config.json (Git: bc20ff2)
	pytorch_model.bin (LFS: 35686c2)
	sentencepiece.bpe.model (LFS: 988bc5a)
	special_tokens_map.json (Git: cb23931)
	tokenizer.json (Git: 851ff3e)
	tokenizer_config.json (Git: f0f7783)

Objects not staged for commit:


```

Podemos ver que todos los archivos tienen `Git` como manejador, excepto *pytorch_model.bin* y *sentencepiece.bpe.model*, que tienen `LFS`. Genial!

**TensorFlow/Keras:**

```bash
On branch main
Objects to be pushed to origin/main:


Objects to be committed:

	config.json (Git: bc20ff2)
	sentencepiece.bpe.model (LFS: 988bc5a)
	special_tokens_map.json (Git: cb23931)
	tf_model.h5 (LFS: 86fce29)
	tokenizer.json (Git: 851ff3e)
	tokenizer_config.json (Git: f0f7783)

Objects not staged for commit:


```

Podemos ver que todos los archivos tienen `Git` como manejador, excepto *t5_model.h5*, que tiene `LFS`. Genial!


Procedamos a los pasos finales, haciendo commit y subiendo al repositorio remoto de *huggingface.co*:

```bash
git commit -m "First model version"
```


**PyTorch:**

```bash
[main b08aab1] First model version
 7 files changed, 29027 insertions(+)
  6 files changed, 36 insertions(+)
 create mode 100644 config.json
 create mode 100644 pytorch_model.bin
 create mode 100644 sentencepiece.bpe.model
 create mode 100644 special_tokens_map.json
 create mode 100644 tokenizer.json
 create mode 100644 tokenizer_config.json
```

**TensorFlow/Keras:**

```bash
[main b08aab1] First model version
 6 files changed, 36 insertions(+)
 create mode 100644 config.json
 create mode 100644 sentencepiece.bpe.model
 create mode 100644 special_tokens_map.json
 create mode 100644 tf_model.h5
 create mode 100644 tokenizer.json
 create mode 100644 tokenizer_config.json
```


Subir puede tomar un poco de tiempo, dependiendo de la velocidad de tu conexion a internet y el tamano de tus archivos:

```bash
git push
```

```bash
Uploading LFS objects: 100% (1/1), 433 MB | 1.3 MB/s, done.
Enumerating objects: 11, done.
Counting objects: 100% (11/11), done.
Delta compression using up to 12 threads
Compressing objects: 100% (9/9), done.
Writing objects: 100% (9/9), 288.27 KiB | 6.27 MiB/s, done.
Total 9 (delta 1), reused 0 (delta 0), pack-reused 0
To https://huggingface.co/lysandre/dummy
   891b41d..b08aab1  main -> main
```


**PyTorch:**

Si echamos un vistazo al repositorio del modelo cuando esto termine, podemos ver todos los archivos agregados recientemente:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/full_model.png" alt="La pestana 'Files and versions' ahora contiene todos los archivos subidos recientemente." width="80%"/>
</div>

La interfaz de usuario te permite explorar los archivos del modelo y commits y ver el diff introducido por cada commit:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/diffs.gif" alt="El diff introducido por el commit reciente." width="80%"/>
</div>

**TensorFlow/Keras:**

Si echamos un vistazo al repositorio del modelo cuando esto termine, podemos ver todos los archivos agregados recientemente:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/full_model_tf.png" alt="La pestana 'Files and versions' ahora contiene todos los archivos subidos recientemente." width="80%"/>
</div>

La interfaz de usuario te permite explorar los archivos del modelo y commits y ver el diff introducido por cada commit:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter4/diffstf.gif" alt="El diff introducido por el commit reciente." width="80%"/>
</div>



---

# Construyendo una tarjeta de modelo[[building-a-model-card]]


La tarjeta de modelo es un archivo que es posiblemente tan importante como los archivos del modelo y tokenizador en un repositorio de modelo. Es la definicion central del modelo, asegurando la reutilizacion por otros miembros de la comunidad y la reproducibilidad de resultados, y proporcionando una plataforma sobre la cual otros miembros pueden construir sus artefactos.

Documentar el proceso de entrenamiento y evaluacion ayuda a otros a entender que esperar de un modelo — y proporcionar informacion suficiente sobre los datos que fueron usados y el preprocesamiento y postprocesamiento que se hicieron asegura que las limitaciones, sesgos y contextos en los que el modelo es y no es util puedan ser identificados y entendidos.

Por lo tanto, crear una tarjeta de modelo que defina claramente tu modelo es un paso muy importante. Aqui, proporcionamos algunos consejos que te ayudaran con esto. Crear la tarjeta de modelo se hace a traves del archivo *README.md* que viste antes, que es un archivo Markdown.

El concepto de "tarjeta de modelo" se origina de una direccion de investigacion de Google, compartida primero en el articulo ["Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993) por Margaret Mitchell et al. Mucha de la informacion contenida aqui esta basada en ese articulo, y te recomendamos echarle un vistazo para entender por que las tarjetas de modelo son tan importantes en un mundo que valora la reproducibilidad, reutilizacion y equidad.

La tarjeta de modelo usualmente comienza con una descripcion muy breve y de alto nivel de para que es el modelo, seguida de detalles adicionales en las siguientes secciones:

- Descripcion del modelo
- Usos previstos y limitaciones
- Como usar
- Limitaciones y sesgo
- Datos de entrenamiento
- Procedimiento de entrenamiento
- Resultados de evaluacion

Veamos que deberia contener cada una de estas secciones.

### Descripcion del modelo[[model-description]]

La descripcion del modelo proporciona detalles basicos sobre el modelo. Esto incluye la arquitectura, version, si fue introducido en un articulo, si hay una implementacion original disponible, el autor e informacion general sobre el modelo. Cualquier derecho de autor deberia atribuirse aqui. Informacion general sobre procedimientos de entrenamiento, parametros y avisos importantes tambien puede mencionarse en esta seccion.

### Usos previstos y limitaciones[[intended-uses-limitations]]

Aqui describes los casos de uso para los que el modelo esta destinado, incluyendo los idiomas, campos y dominios donde puede aplicarse. Esta seccion de la tarjeta de modelo tambien puede documentar areas que se sabe que estan fuera del alcance del modelo, o donde es probable que tenga un rendimiento suboptimo.

### Como usar[[how-to-use]]

Esta seccion deberia incluir algunos ejemplos de como usar el modelo. Esto puede mostrar el uso de la funcion `pipeline()`, uso de las clases de modelo y tokenizador, y cualquier otro codigo que creas que podria ser util.

### Datos de entrenamiento[[training-data]]

Esta parte deberia indicar en que conjunto(s) de datos fue entrenado el modelo. Una breve descripcion del conjunto o conjuntos de datos tambien es bienvenida.

### Procedimiento de entrenamiento[[training-procedure]]

En esta seccion deberias describir todos los aspectos relevantes del entrenamiento que son utiles desde una perspectiva de reproducibilidad. Esto incluye cualquier preprocesamiento y postprocesamiento que se hizo en los datos, asi como detalles como el numero de epocas por las que el modelo fue entrenado, el tamano del lote, la tasa de aprendizaje, y asi sucesivamente.

### Variables y metricas[[variable-and-metrics]]

Aqui deberias describir las metricas que usas para evaluacion, y los diferentes factores que estas midiendo. Mencionar que metrica(s) fueron usadas, en que conjunto de datos y que division del conjunto de datos, facilita comparar el rendimiento de tu modelo con el de otros modelos. Esto deberia estar informado por las secciones anteriores, como los usuarios previstos y casos de uso.

### Resultados de evaluacion[[evaluation-results]]

Finalmente, proporciona una indicacion de que tan bien se desempena el modelo en el conjunto de datos de evaluacion. Si el modelo usa un umbral de decision, proporciona el umbral de decision usado en la evaluacion, o proporciona detalles sobre la evaluacion en diferentes umbrales para los usos previstos.

## Ejemplo[[example]]

Consulta los siguientes ejemplos de tarjetas de modelo bien elaboradas:

- [`bert-base-cased`](https://huggingface.co/bert-base-cased)
- [`gpt2`](https://huggingface.co/gpt2)
- [`distilbert`](https://huggingface.co/distilbert-base-uncased)

Mas ejemplos de diferentes organizaciones y companias estan disponibles [aqui](https://github.com/huggingface/model_card/blob/master/examples.md).

## Nota[[note]]

Las tarjetas de modelo no son un requisito al publicar modelos, y no necesitas incluir todas las secciones descritas arriba cuando hagas una. Sin embargo, documentacion explicita del modelo solo puede beneficiar a futuros usuarios, por lo que recomendamos que completes tantas secciones como sea posible con lo mejor de tu conocimiento y capacidad.

## Metadatos de la tarjeta de modelo[[model-card-metadata]]

Si has explorado un poco el Hub de Hugging Face, deberias haber visto que algunos modelos pertenecen a ciertas categorias: puedes filtrarlos por tareas, idiomas, bibliotecas y mas. Las categorias a las que pertenece un modelo se identifican segun los metadatos que agregas en el encabezado de la tarjeta de modelo.

Por ejemplo, si echas un vistazo a la [tarjeta de modelo `camembert-base`](https://huggingface.co/camembert-base/blob/main/README.md), deberias ver las siguientes lineas en el encabezado de la tarjeta de modelo:

```
---
language: fr
license: mit
datasets:
- oscar
---
```

Estos metadatos son analizados por el Hub de Hugging Face, que luego identifica este modelo como un modelo frances, con una licencia MIT, entrenado en el conjunto de datos Oscar.

La [especificacion completa de la tarjeta de modelo](https://github.com/huggingface/hub-docs/blame/main/modelcard.md) permite especificar idiomas, licencias, etiquetas, conjuntos de datos, metricas, asi como los resultados de evaluacion que el modelo obtuvo durante el entrenamiento.


---

# Parte 1 completada![[part-1-completed]]


Este es el final de la primera parte del curso! La Parte 2 sera lanzada el 15 de noviembre con un gran evento comunitario, consulta mas informacion [aqui](https://huggingface.co/blog/course-launch-event).

Ahora deberias ser capaz de ajustar finamente un modelo preentrenado en un problema de clasificacion de texto (oraciones individuales o pares de oraciones) y subir el resultado al Model Hub. Para asegurarte de que dominaste esta primera seccion, deberias hacer exactamente eso en un problema que te interese (y no necesariamente en ingles si hablas otro idioma)! Puedes encontrar ayuda en los [foros de Hugging Face](https://discuss.huggingface.co/) y compartir tu proyecto en [este tema](https://discuss.huggingface.co/t/share-your-projects/6803) una vez que hayas terminado.

No podemos esperar a ver lo que construiras con esto!


---




# Cuestionario de fin de capitulo[[end-of-chapter-quiz]]


Probemos lo que aprendiste en este capitulo!

### 1. A que estan limitados los modelos en el Hub?


- Modelos de la biblioteca 🤗 Transformers.
- Todos los modelos con una interfaz similar a 🤗 Transformers.
- No hay limites.
- Modelos que estan de alguna manera relacionados con NLP.


### 2. Como puedes administrar modelos en el Hub?


- A traves de una cuenta de GCP.
- A traves de distribucion peer-to-peer.
- A traves de git y git-lfs.


### 3. Que puedes hacer usando la interfaz web del Hub de Hugging Face?


- Hacer fork de un repositorio existente.
- Crear un nuevo repositorio de modelo.
- Administrar y editar archivos.
- Subir archivos.
- Ver diffs entre versiones.


### 4. Que es una tarjeta de modelo?


- Una descripcion aproximada del modelo, por lo tanto menos importante que los archivos del modelo y tokenizador.
- Una forma de asegurar reproducibilidad, reutilizacion y equidad.
- Un archivo Python que puede ejecutarse para recuperar informacion sobre el modelo.


### 5. Cuales de estos objetos de la biblioteca 🤗 Transformers pueden compartirse directamente en el Hub con `push_to_hub()`?


**PyTorch:**


- Un tokenizador
- Una configuracion de modelo
- Un modelo
- Un Trainer


**TensorFlow/Keras:**


- Un tokenizador
- Una configuracion de modelo
- Un modelo
- Todo lo anterior con un callback dedicado


### 6. Cual es el primer paso al usar el metodo `push_to_hub()` o las herramientas CLI?


- Iniciar sesion en el sitio web.
- Ejecutar 
- Ejecutar 


### 7. Estas usando un modelo y un tokenizador — como puedes subirlos al Hub?


- Llamando al metodo push_to_hub directamente en el modelo y el tokenizador.
- Dentro del entorno de ejecucion de Python, envolviendolos en una utilidad de `huggingface_hub`.
- Guardandolos en disco y llamando a `transformers-cli upload-model`


### 8. Que operaciones de git puedes hacer con la clase `Repository`?


- Un commit.
- Un pull
- Un push
- Un merge



---

# 5. La librería 🤗 Datasets

# Introducción


En el [Capítulo 3](/course/chapter3) tuviste tu primer acercamiento a la librería 🤗 Datasets y viste que existían 3 pasos principales para ajustar un modelo:

1. Cargar un conjunto de datos del Hub de Hugging Face.
2. Preprocesar los datos con `Dataset.map()`.
3. Cargar y calcular métricas.

¡Esto es apenas el principio de lo que 🤗 Datasets puede hacer! En este capítulo vamos a estudiar a profundidad esta librería y responderemos las siguientes preguntas:

* ¿Qué hacer cuando tu dataset no está en el Hub?
* ¿Cómo puedes subdividir tu dataset? (¿Y qué hacer si _realmente_ necesitas usar Pandas?)
* ¿Qué hacer cuando tu dataset es enorme y consume toda la RAM de tu computador?
* ¿Qué es la proyección en memoria (_memory mapping_) y Apache Arrow?
* ¿Cómo puedes crear tu propio dataset y subirlo al Hub?

Las técnicas que aprenderás aquí te van a preparar para las tareas de _tokenización_ avanzada y ajuste que verás en el [Capítulo 6](/course/chapter6) y el [Capítulo 7](/course/chapter7). ¡Así que ve por un café y arranquemos!

---

# ¿Y si mi dataset no está en el Hub?


Ya sabes cómo usar el [Hub de Hugging Face](https://huggingface.co/datasets) para descargar datasets, pero usualmente vas a tener que trabajar con datos que están guardados en tu computador o en un servidor remoto. En esta sección te mostraremos cómo usar 🤗 Datasets para cargar conjuntos de datos que no están disponibles en el Hub de Hugging Face.


**Video:** [Ver en YouTube](https://youtu.be/HyQgpJTkRdE)


## Trabajando con datos locales y remotos

🤗 Datasets contiene scripts para cargar datasets locales y remotos que soportan formatos comunes de datos como:

|    Formato de datos     | Script de carga |                         Ejemplo                         |
| :----------------: | :------------: | :-----------------------------------------------------: |
|     CSV y TSV      |     `csv`      |     `load_dataset("csv", data_files="my_file.csv")`     |
|     Archivos de texto     |     `text`     |    `load_dataset("text", data_files="my_file.txt")`     |
| JSON y JSON Lines  |     `json`     |   `load_dataset("json", data_files="my_file.jsonl")`    |
| Pickled DataFrames |    `pandas`    | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

Como ves en la tabla, para cada formato de datos solo tenemos que especificar el tipo de script de carga en la función `load_dataset()`, así como el argumento `data_files` que contiene la ruta a uno o más archivos. Comencemos por cargar un dataset desde archivos locales y luego veremos cómo hacer lo propio para archivos remotos.

## Cargando un dataset local

Para este ejemplo, vamos a usar el [dataset SQuAD-it], que es un dataset de gran escala para responder preguntas en italiano.

Los conjuntos de entrenamiento y de prueba están alojados en GitHub, así que podemos descargarlos fácilmente con el comando `wget`:

```python
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
```

Esto va a descargar dos archivos comprimidos llamados *SQuAD_it-train.json.gz* y *SQuAD_it-test.json.gz*, que podemos descomprimir con el comando  `gzip` de Linux:

```python
!gzip -dkv SQuAD_it-*.json.gz
```

```bash
SQuAD_it-test.json.gz:	   87.4% -- replaced with SQuAD_it-test.json
SQuAD_it-train.json.gz:	   82.2% -- replaced with SQuAD_it-train.json
```

De este modo, podemos ver que los archivos comprimidos son reemplazados por los archuvos en formato JSON _SQuAD_it-train.json_ y _SQuAD_it-test.json_.

> [!TIP]
> ✎ Si te preguntas por qué hay un carácter de signo de admiración (`!`) en los comandos de shell, esto es porque los estamos ejecutando desde un cuaderno de Jupyter. Si quieres descargar y descomprimir el archivo directamente desde la terminal, elimina el signo de admiración.

Para cargar un archivo JSON con la función `load_dataset()`, necesitamos saber si estamos trabajando con un archivo JSON ordinario (parecido a un diccionario anidado) o con JSON Lines (JSON separado por líneas). Como muchos de los datasets de respuesta a preguntas que te vas a encontrar, SQuAD-it usa el formato anidado, en el que el texto está almacenado en un campo `data`. Esto significa que podemos cargar el dataset especificando el argumento `field` de la siguiente manera: 

```py
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
```

Por defecto, cuando cargas archivos locales se crea un objeto `DatasetDict` con un conjunto de entrenamiento –`train`–. Podemos verlo al inspeccionar el objeto `squad_it_dataset`:

```py
squad_it_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
```

Esto nos muestra el número de filas y los nombres de las columnas asociadas al conjunto de entrenamiento. Podemos ver uno de los ejemplos al poner un índice en el conjunto de entrenamiento así:

```py
squad_it_dataset["train"][0]
```

```python out
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}
```

¡Genial, ya cargamos nuestro primer dataset local! Sin embargo, esto funcionó únicamente para el conjunto de entrenamiento. Realmente, queremos incluir tanto el conjunto `train` como el conjunto `test` en un único objeto `DatasetDict` para poder aplicar las funciones `Dataset.map()` en ambos conjuntos al mismo tiempo. Para hacerlo, podemos incluir un diccionario en el argumento `datafiles` que mapea cada nombre de conjunto a su archivo asociado:


```py
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```

Esto es exactamente lo que queríamos. Ahora podemos aplicar varias técnicas de preprocesamiento para limpiar los datos, _tokenizar_ las reseñas, entre otras tareas.

> [!TIP]
> El argumento `data_files` de la función `load_dataset()` es muy flexible. Puede ser una única ruta de archivo, una lista de rutas o un diccionario que mapee los nombres de los conjuntos a las rutas de archivo. También puedes buscar archivos que cumplan con cierto patrón específico de acuerdo con las reglas usadas por el shell de Unix (e.g., puedes buscar todos los archivos JSON en una carpeta al definir `datafiles="*.json"`). Revisa la [documentación](https://huggingface.co/docs/datasets/loading#local-and-remote-files) para más detalles.

Los scripts de carga en 🤗 Datasets también pueden descomprimir los archivos de entrada automáticamente, así que podemos saltarnos el uso de `gzip` especificando el argumento `data_files` directamente a la ruta de los archivos comprimidos.

```py
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

Esto puede ser útil si no quieres descomprimir manualmente muchos archivos GZIP. La descompresión automática también aplica para otros formatos de archivo comunes como TAR y ZIP, así que solo necesitas dirigir el argumento `data_files` a los archivos comprimidos y ¡listo!.

Ahora que sabes cómo cargar archivos locales en tu computador portátil o de escritorio, veamos cómo cargar archivos remotos.

## Cargando un dataset remoto

Si estás trabajando como científico de datos o desarrollador en una compañía, hay una alta probabilidad de que los datasets que quieres analizar estén almacenados en un servidor remoto. Afortunadamente, ¡la carga de archivos remotos es tan fácil como cargar archivos locales! En vez de incluir una ruta de archivo, dirigimos el argumento `data_files` de la función `load_datasets()` a una o más URL en las que estén almacenados los archivos. Por ejemplo, para el dataset SQuAD-it alojado en GitHub, podemos apuntar `data_files` a las URL de _SQuAD_it-*.json.gz_ así:

```py
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

Esto devuelve el mismo objeto `DatasetDict` que obtuvimos antes, pero nos ahorra el paso de descargar y descomprimir manualmente los archivos _SQuAD_it-*.json.gz_. Con esto concluimos nuestra exploración de las diferentes maneras de cargar datasets que no están alojados en el Hub de Hugging Face. Ahora que tenemos un dataset para experimentar, ¡pongámonos manos a la obra con diferentes técnicas de procesamiento de datos!

> [!TIP]
> ✏️ **¡Inténtalo!** Escoge otro dataset alojado en GitHub o en el [Repositorio de Machine Learning de UCI](https://archive.ics.uci.edu/ml/index.php) e intenta cargarlo local y remotamente usando las técnicas descritas con anterioridad. Para puntos extra, intenta cargar un dataset que esté guardado en un formato CSV o de texto (revisa la [documentación](https://huggingface.co/docs/datasets/loading#local-and-remote-files) pata tener más información sobre estos formatos).


---

# Es momento de subdividir


La mayor parte del tiempo tus datos no estarán perfectamente listos para entrenar modelos. En esta sección vamos a explorar distintas funciones que tiene 🤗 Datasets para limpiar tus conjuntos de datos.


**Video:** [Ver en YouTube](https://youtu.be/tqfSFcPMgOI)


## Subdividiendo nuestros datos

De manera similar a Pandas, 🤗 Datasets incluye varias funciones para manipular el contenido de los objetos `Dataset` y `DatasetDict`. Ya vimos el método `Dataset.map()` en el [Capítulo 3](/course/chapter3) y en esta sección vamos a explorar otras funciones que tenemos a nuestra disposición.

Para este ejemplo, vamos a usar el [Dataset de reseñas de medicamentos](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) alojado en el [Repositorio de Machine Learning de UC Irvine](https://archive.ics.uci.edu/ml/index.php), que contiene la evaluación de varios medicamentos por parte de pacientes, junto con la condición por la que los estaban tratando y una calificación en una escala de 10 estrellas sobre su satisfacción.

Primero, tenemos que descargar y extraer los datos, que se puede hacer con los comandos `wget` y `unzip`:

```py
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip
```

Dado que TSV es una variación de CSV en la que se usan tabulaciones en vez de comas como separadores, podemos cargar estos archivos usando el script de carga `csv` y especificando el argumento `delimiter` en la función `load_dataset` de la siguiente manera:

```py
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t es el carácter para tabulaciones en Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
```

Una buena práctica al hacer cualquier tipo de análisis de datos es tomar una muestra aleatoria del dataset para tener una vista rápida del tipo de datos con los que estás trabajando. En 🤗 Datasets, podemos crear una muestra aleatoria al encadenar las funciones `Dataset.shuffle()` y `Dataset.select()`:

```py
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Mirar los primeros ejemplos
drug_sample[:3]
```

```python out
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
```

Puedes ver que hemos fijado la semilla en `Dataset.shuffle()` por motivos de reproducibilidad. `Dataset.select()` espera un iterable de índices, así que incluimos `range(1000)` para tomar los primeros 1.000 ejemplos del conjunto de datos aleatorizado. Ya podemos ver algunos detalles para esta muestra:

* La columna `Unnamed: 0` se ve sospechosamente como un ID anonimizado para cada paciente.
* La columna `condition` incluye una mezcla de niveles en mayúscula y minúscula.
* Las reseñas tienen longitud variable y contienen una mezcla de separadores de línea de Python (`\r\n`), así como caracteres de HTML como `&\#039;`.

Veamos cómo podemos usar 🤗 Datasets para lidiar con cada uno de estos asuntos. Para probar la hipótesis de que la columna `Unnamed: 0` es un ID de los pacientes, podemos usar la función `Dataset.unique()` para verificar que el número de los ID corresponda con el número de filas de cada conjunto:

```py
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
```

Esto parece confirmar nuestra hipótesis, así que limpiemos el dataset un poco al cambiar el nombre de la columna `Unnamed: 0` a algo más legible. Podemos usar la función `DatasetDict.rename_column()` para renombrar la columna en ambos conjuntos en una sola operación:

```py
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
drug_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
```

> [!TIP]
> ✏️ **¡Inténtalo!** Usa la función `Dataset.unique()` para encontrar el número de medicamentos y condiciones únicas en los conjuntos de entrenamiento y de prueba.

Ahora normalicemos todas las etiquetas de `condition` usando `Dataset.map()`. Tal como lo hicimos con la tokenización en el [Capítulo 3](/course/chapter3), podemos definir una función simple que pueda ser aplicada en todas las filas de cada conjunto en el `drug_dataset`:

```py
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset.map(lowercase_condition)
```

```python out
AttributeError: 'NoneType' object has no attribute 'lower'
```

¡Tenemos un problema en nuestra función de mapeo! Del error podemos inferir que algunas de las entradas de la columna `condición` son `None`, que no puede transformarse en minúscula al no ser un string. Filtremos estas filas usando `Dataset.filter()`, que funciona de una forma similar `Dataset.map()` y recibe como argumento una función que toma un ejemplo particular del dataset. En vez de escribir una función explícita como:

```py
def filter_nones(x):
    return x["condition"] is not None
```

y luego ejecutar `drug_dataset.filter(filter_nones)`, podemos hacerlo en una línea usando una _función lambda_. En Python, las funciones lambda son funciones pequeñas que puedes definir sin nombrarlas explícitamente. Estas toman la forma general:

```
lambda <arguments> : <expression>
```

en la que `lambda` es una de las [palabras especiales](https://docs.python.org/3/reference/lexical_analysis.html#keywords) de Python, `<arguments>` es una lista o conjunto de valores separados con coma que definen los argumentos de la función y `<expression>` representa las operaciones que quieres ejecutar. Por ejemplo, podemos definir una función lambda simple que eleve un número al cuadrado de la siguiente manera:

```
lambda x : x * x
```

Para aplicar esta función a un _input_, tenemos que envolverla a ella y al _input_ en paréntesis:

```py
(lambda x: x * x)(3)
```

```python out
9
```

De manera similar, podemos definir funciones lambda con múltiples argumentos separándolos con comas. Por ejemplo, podemos calcular el área de un triángulo así:

```py
(lambda base, height: 0.5 * base * height)(4, 8)
```

```python out
16.0
```

Las funciones lambda son útiles cuando quieres definir funciones pequeñas de un único uso (para más información sobre ellas, te recomendamos leer este excelente [tutorial de Real Python](https://realpython.com/python-lambda/) escrito por Andre Burgaud). En el contexto de 🤗 Datasets, podemos usar las funciones lambda para definir operaciones simples de mapeo y filtrado, así que usemos este truco para eliminar las entradas `None` de nuestro dataset:

```py
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
```

Ahora que eliminamos los `None`, podemos normalizar nuestra columna `condition`:

```py
drug_dataset = drug_dataset.map(lowercase_condition)
# Revisar que se pasaron a minúscula
drug_dataset["train"]["condition"][:3]
```

```python out
['left ventricular dysfunction', 'adhd', 'birth control']
```

¡Funcionó! Como ya limpiamos las etiquetas, veamos cómo podemos limpiar las reseñas.

## Creando nuevas columnas

Cuando estás lidiando con reseñas de clientes, es una buena práctica revisar el número de palabras de cada reseña. Una reseña puede ser una única palabra como "¡Genial!" o un ensayo completo con miles de palabras y, según el caso de uso, tendrás que abordar estos extremos de forma diferente. Para calcular el número de palabras en cada reseña, usaremos una heurística aproximada basada en dividir cada texto por los espacios en blanco.

Definamos una función simple que cuente el número de palabras en cada reseña:

```py
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
```

Contrario a la función `lowercase_condition()`, `compute_review_length()` devuelve un diccionario cuya llave no corresponde a uno de los nombres de las columnas en el conjunto de datos. En este caso, cuando se pasa `compute_review_length()` a `Dataset.map()`,  la función se aplicará a todas las filas en el dataset para crear una nueva columna `review_length()`:

```py
drug_dataset = drug_dataset.map(compute_review_length)
# Inspeccionar el primer ejemplo de entrenamiento
drug_dataset["train"][0]
```

```python out
{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
```

Tal como lo esperábamos, podemos ver que se añadió la columna `review_length` al conjunto de entrenamiento. Podemos ordenar esta columna nueva con `Dataset.sort()` para ver cómo son los valores extremos:

```py
drug_dataset["train"].sort("review_length")[:3]
```

```python out
{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}
```

Como lo discutimos anteriormente, algunas reseñas incluyen una sola palabra, que si bien puede ser útil para el análisis de sentimientos, no sería tan informativa si quisiéramos predecir la condición.

> [!TIP]
> 🙋 Una forma alternativa de añadir nuevas columnas al dataset es a través de la función `Dataset.add_column()`. Esta te permite incluir la columna como una lista de Python o un array de NumPy y puede ser útil en situaciones en las que `Dataset.map()` no se ajusta a tu caso de uso.

Usemos la función `Dataset.filter()` para quitar las reseñas que contienen menos de 30 palabras. Similar a lo que hicimos con la columna `condition`, podemos filtrar las reseñas cortas al incluir una condición de que su longitud esté por encima de este umbral:

```py
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)
```

```python out
{'train': 138514, 'test': 46108}
```

Como puedes ver, esto ha eliminado alrededor del 15% de las reseñas de nuestros conjuntos originales de entrenamiento y prueba.

> [!TIP]
> ✏️ **¡Inténtalo!** Usa la función `Dataset.sort()` para inspeccionar las reseñas con el mayor número de palabras. Revisa la [documentación](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.sort) para ver cuál argumento necesitas para ordenar las reseñas de mayor a menor.

Por último, tenemos que lidiar con la presencia de códigos de caracteres HTML en las reseñas. Podemos usar el módulo `html` de Python para transformar estos códigos así:

```py
import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)
```

```python out
"I'm a transformer called BERT"
```

Usaremos `Dataset.map()` para transformar todos los caracteres HTML en el corpus:

```python
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
```

Como puedes ver, el método `Dataset.map()` es muy útil para procesar datos y esta es apenas la punta del iceberg de lo que puede hacer.

## Los superpoderes del método `map()`

El método `Dataset.map()` recibe un argumento `matched` que, al definirse como `True`, envía un lote de ejemplos a la función de mapeo a la vez (el tamaño del lote se puede configurar, pero tiene un valor por defecto de 1.000). Por ejemplo, la función anterior de mapeo que transformó todos los HTML se demoró un poco en su ejecución (puedes leer el tiempo en las barras de progreso). Podemos reducir el tiempo al procesar varios elementos a la vez usando un _list comprehension_.

Cuando especificas `batched=True`, la función recibe un diccionario con los campos del dataset, pero cada valor es ahora una _lista de valores_ y no un valor individual. La salida de `Dataset.map()` debería ser igual: un diccionario con los campos que queremos actualizar o añadir a nuestro dataset y una lista de valores. Por ejemplo, aquí puedes ver otra forma de transformar todos los caracteres HTML usando `batched=True`:

```python
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
```

Si estás ejecutando este código en un cuaderno, verás que este comando se ejecuta mucho más rápido que el anterior. Y no es porque los caracteres HTML de las reseñas ya se hubieran procesado; si vuelves a ejecutar la instrucción de la sección anterior (sin `batched=True`), se tomará el mismo tiempo de ejecución que antes. Esto es porque las _list comprehensions_ suelen ser más rápidas que ejecutar el mismo código en un ciclo `for` y porque también ganamos rendimiento al acceder a muchos elementos a la vez en vez de uno por uno.

Usar `Dataset.map()` con `batched=True` será fundamental para desbloquear la velocidad de los tokenizadores "rápidos" que nos vamos a encontrar en el [Capítulo 6](/course/chapter6), que pueden tokenizar velozmente grandes listas de textos. Por ejemplo, para tokenizar todas las reseñas de medicamentos con un tokenizador rápido, podríamos usar una función como la siguiente:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)
```

Como viste en el [Capítulo 3](/course/chapter3), podemos pasar uno o varios ejemplos al tokenizador, así que podemos usar esta función con o sin `batched=True`. Aprovechemos esta oportunidad para comparar el desempeño de las distintas opciones. En un cuaderno, puedes medir el tiempo de ejecución de una instrucción de una línea añadiendo `%time` antes de la línea de código de tu interés:

```python no-format
%time tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
```

También puedes medir el tiempo de una celda completa añadiendo `%%time` al inicio de la celda. En el hardware en el que lo ejecutamos, nos arrojó 10.8s para esta instrucción (es el número que aparece después de "Wall time").

> [!TIP]
> ✏️ **¡Inténtalo!** Ejecuta la misma instrucción con y sin `batched=True` y luego usa un tokenizador "lento" (añade `use_fast=False` en el método `AutoTokenizer.from_pretrained()`) para ver cuánto tiempo se toman en tu computador.

Estos son los resultados que obtuvimos con y sin la ejecución por lotes, con un tokenizador rápido y lento:

Opciones         | Tokenizador rápido | Tokenizador lento
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s

Esto significa que usar un tokenizador rápido con la opción `batched=True` es 30 veces más rápido que su contraparte lenta sin usar lotes. ¡Realmente impresionante! Esta es la razón principal por la que los tokenizadores rápidos son la opción por defecto al usar `AutoTokenizer` (y por qué se denominan "rápidos"). Estos logran tal rapidez gracias a que el código de los tokenizadores corre en Rust, que es un lenguaje que facilita la ejecución del código en paralelo.

La paralelización también es la razón para el incremento de 6x en la velocidad del tokenizador al ejecutarse por lotes: No puedes ejecutar una única operación de tokenización en paralelo, pero cuando quieres tokenizar muchos textos al mismo tiempo puedes dividir la ejecución en diferentes procesos, cada uno responsable de sus propios textos.

`Dataset.map()` también tiene algunas capacidades de paralelización. Dado que no funcionan con Rust, no van a hacer que un tokenizador lento alcance el rendimiento de uno rápido, pero aún así pueden ser útiles (especialmente si estás usando un tokenizador que no tiene una versión rápida). Para habilitar el multiprocesamiento, usa el argumento `num_proc` y especifica el número de procesos para usar en `Dataset.map()`:

```py
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```

También puedes medir el tiempo para determinar el número de procesos que vas a usar. En nuestro caso, usar 8 procesos produjo la mayor ganancia de velocidad. Aquí están algunos de los números que obtuvimos con y sin multiprocesamiento:

Opciones         | Tokenizador rápido | Tokenizador lento
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s
`batched=True`, `num_proc=8`  | 6.52s          | 41.3s
`batched=False`, `num_proc=8` | 9.49s          | 45.2s

Estos son resultados mucho más razonables para el tokenizador lento, aunque el desempeño del rápido también mejoró sustancialmente. Sin embargo, este no siempre será el caso: para valores de `num_proc` diferentes a 8, nuestras pruebas mostraron que era más rápido usar `batched=true` sin esta opción. En general, no recomendamos usar el multiprocesamiento de Python para tokenizadores rápidos con `batched=True`.

> [!TIP]
> Usar `num_proc` para acelerar tu procesamiento suele ser una buena idea, siempre y cuando la función que uses no esté usando multiples procesos por si misma.

Que toda esta funcionalidad está incluida en un método es algo impresionante en si mismo, ¡pero hay más!. Con `Dataset.map()` y `batched=True` puedes cambiar el número de elementos en tu dataset. Esto es súper útil en situaciones en las que quieres crear varias características de entrenamiento de un ejemplo, algo que haremos en el preprocesamiento para varias de las tareas de PLN que abordaremos en el [Capítulo 7](/course/chapter7).

> [!TIP]
> 💡 Un _ejemplo_ en Machine Learning se suele definir como el conjunto de _features_ que le damos al modelo. En algunos contextos estos features serán el conjunto de columnas en un `Dataset`, mientras que en otros se pueden extraer múltiples features de un solo ejemplo que pertenecen a una columna –como aquí y en tareas de responder preguntas-.

¡Veamos cómo funciona! En este ejemplo vamos a tokenizar nuestros ejemplos y limitarlos a una longitud máxima de 128, pero le pediremos al tokenizador que devuelva *todos* los fragmentos de texto en vez de unicamente el primero. Esto se puede lograr con el argumento `return_overflowing_tokens=True`:

```py
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
```

Probémoslo en un ejemplo puntual antes de usar `Dataset.map()` en todo el dataset:

```py
result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]
```

```python out
[128, 49]
```

El primer ejemplo en el conjunto de entrenamiento se convirtió en dos features porque fue tokenizado en un número superior de tokens al que especificamos: el primero de longitud 128 y el segundo de longitud 49. ¡Vamos a aplicarlo a todo el dataset!

```py
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
```

```python out
ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000
```

¿Por qué no funcionó? El mensaje de error nos da una pista: hay un desajuste en las longitudes de una de las columnas, siendo una de longitud 1.463 y otra de longitud 1.000. Si has revisado la [documentación de `Dataset.map()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map), te habrás dado cuenta que estamos mapeando el número de muestras que le pasamos a la función: en este caso los 1.000 ejemplos nos devuelven 1.463 features, arrojando un error.

El problema es que estamos tratando de mezclar dos datasets de tamaños diferentes: las columnas de `drug_dataset` tendrán un cierto número de ejemplos (los 1.000 en el error), pero el `tokenized_dataset` que estamos construyendo tendrá más (los 1.463 en el mensaje de error). Esto no funciona para un `Dataset`, así que tenemos que eliminar las columnas del anterior dataset o volverlas del mismo tamaño del nuevo. Podemos hacer la primera operación con el argumento `remove_columns`:

```py
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
```

Ahora funciona sin errores. Podemos revisar que nuestro dataset nuevo tiene más elementos que el original al comparar sus longitudes:

```py
len(tokenized_dataset["train"]), len(drug_dataset["train"])
```

```python out
(206772, 138514)
```

También mencionamos que podemos trabajar con el problema de longitudes que no coinciden al convertir las columnas viejas en el mismo tamaño de las nuevas. Para eso, vamos a necesitar el campo `overflow_to_sample_mapping` que devuelve el tokenizer cuando definimos `return_overflowing_tokens=True`. Esto devuelve un mapeo del índice de un nuevo feature al índice de la muestra de la que se originó. Usando lo anterior, podemos asociar cada llave presente en el dataset original con una lista de valores del tamaño correcto al repetir los valores de cada ejemplo tantas veces como genere nuevos features:

```py
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extraer el mapeo entre los índices nuevos y viejos
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
```

De esta forma, podemos ver que funciona con `Dataset.map()` sin necesidad de eliminar las columnas viejas.

```py
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
```

Como resultado, tenemos el mismo número de features de entrenamiento que antes, pero conservando todos los campos anteriores. Quizás prefieras usar esta opción si necesitas conservarlos para algunas tareas de post-procesamiento después de aplicar tu modelo.

Ya has visto como usar 🤗 Datasets para preprocesar un dataset de varias formas. Si bien las funciones de procesamiento de 🤗 Datasets van a suplir la mayor parte de tus necesidades de entrenamiento de modelos, hay ocasiones en las que puedes necesitar Pandas para tener acceso a herramientas más poderosas, como `DataFrame.groupby()` o algún API de alto nivel para visualización. Afortunadamente, 🤗 Datasets está diseñado para ser interoperable con librerías como Pandas, NumPy, PyTorch, TensoFlow y JAX. Veamos cómo funciona.

## De `Dataset`s a `DataFrame`s y viceversa


**Video:** [Ver en YouTube](https://youtu.be/tfcY1067A5Q)


Para habilitar la conversión entre varias librerías de terceros, 🤗 Datasets provee la función `Dataset.set_format()`. Esta función sólo cambia el _formato de salida_ del dataset, de tal manera que puedas cambiar a otro formato sin cambiar el _formato de datos subyacente_, que es Apache Arrow. Este cambio de formato se hace _in place_. Para verlo en acción, convirtamos el dataset a Pandas: 

```py
drug_dataset.set_format("pandas")
```

Ahora, cuando accedemos a los elementos del dataset obtenemos un `pandas.DataFrame` en vez de un diccionario:

```py
drug_dataset["train"][:3]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>drugName</th>
      <th>condition</th>
      <th>review</th>
      <th>rating</th>
      <th>date</th>
      <th>usefulCount</th>
      <th>review_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95260</td>
      <td>Guanfacine</td>
      <td>adhd</td>
      <td>"My son is halfway through his fourth week of Intuniv..."</td>
      <td>8.0</td>
      <td>April 27, 2010</td>
      <td>192</td>
      <td>141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>92703</td>
      <td>Lybrel</td>
      <td>birth control</td>
      <td>"I used to take another oral contraceptive, which had 21 pill cycle, and was very happy- very light periods, max 5 days, no other side effects..."</td>
      <td>5.0</td>
      <td>December 14, 2009</td>
      <td>17</td>
      <td>134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138000</td>
      <td>Ortho Evra</td>
      <td>birth control</td>
      <td>"This is my first time using any form of birth control..."</td>
      <td>8.0</td>
      <td>November 3, 2015</td>
      <td>10</td>
      <td>89</td>
    </tr>
  </tbody>
</table>

Creemos un `pandas.DataFrame` para el conjunto de entrenamiento entero al seleccionar los elementos de `drug_dataset["train"]`:

```py
train_df = drug_dataset["train"][:]
```

> [!TIP]
> 🚨 Internamente, `Dataset.set_format()` cambia el formato de devolución del método _dunder_ `__getitem()__`. Esto significa que cuando queremos crear un objeto nuevo como `train_df` de un `Dataset` en formato `"pandas"`, tenemos que seleccionar el dataset completo para obtener un `pandas.DataFrame`. Puedes verificar por ti mismo que el tipo de `drug_dataset["train"]` es `Dataset` sin importar el formato de salida.

De aquí en adelante podemos usar toda la funcionalidad de pandas cuando queramos. Por ejemplo, podemos hacer un encadenamiento sofisticado para calcular la distribución de clase entre las entradas de `condition`:

```py
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
frequencies.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>birth control</td>
      <td>27655</td>
    </tr>
    <tr>
      <th>1</th>
      <td>depression</td>
      <td>8023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>acne</td>
      <td>5209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>anxiety</td>
      <td>4991</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pain</td>
      <td>4744</td>
    </tr>
  </tbody>
</table>

Y una vez hemos concluido el análisis con Pandas, tenemos la posibilidad de crear un nuevo objeto `Dataset` usando la función `Dataset.from_pandas()` de la siguiente manera:

```py
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
freq_dataset
```

```python out
Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
```

> [!TIP]
> ✏️ **¡Inténtalo!** Calcula la calificación promedio por medicamento y guarda el resultado en un nuevo `Dataset`.

Con esto terminamos nuestro tour de las múltiples técnicas de preprocesamiento disponibles en 🤗 Datasets. Para concluir, creemos un set de validación para preparar el conjunto de datos y entrenar el clasificador. Antes de hacerlo, vamos a reiniciar el formato de salida de `drug_dataset` de `"pandas"` a `"arrow"`:

```python
drug_dataset.reset_format()
```

## Creando un conjunto de validación

Si bien tenemos un conjunto de prueba que podríamos usar para la evaluación, es una buena práctica dejar el conjunto de prueba intacto y crear un conjunto de validación aparte durante el desarrollo. Una vez estés satisfecho con el desempeño de tus modelos en el conjunto de validación, puedes hacer un último chequeo con el conjunto de prueba. Este proceso ayuda a reducir el riesgo de sobreajustar al conjunto de prueba y desplegar un modelo que falle en datos reales.

🤗 Datasets provee la función `Dataset.train_test_split()` que está basada en la famosa funcionalidad de `scikit-learn`. Usémosla para separar nuestro conjunto de entrenamiento en dos partes `train` y `validation` (definiendo el argumento `seed` por motivos de reproducibilidad):

```py
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Renombrar el conjunto "test" a "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Añadir el conjunto "test" al `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 46108
    })
})
```

Súper, ya preparamos un dataset que está listo para entrenar modelos. En la [sección 5](/course/chapter5/5) veremos cómo subir datasets al Hub de Hugging Face, pero por ahora terminemos el análisis estudiando algunas formas de guardarlos en tu máquina local.

## Saving a dataset


**Video:** [Ver en YouTube](https://youtu.be/blF9uxYcKHo)


A pesar de que 🤗 Datasets va a guardar en caché todo dataset que descargues, así como las operaciones que se ejecutan en él, hay ocasiones en las que querrás guardar un dataset en memoria (e.g., en caso que el caché se elimine). Como se muestra en la siguiente tabla, 🤗 Datasets tiene 3 funciones para guardar tu dataset en distintos formatos:


| Formato |        Función        |
| :---------: | :--------------------: |
|    Arrow    | `Dataset.save_to_disk()` |
|     CSV     |    `Dataset.to_csv()`    |
|    JSON     |   `Dataset.to_json()`    |

Por ejemplo, guardemos el dataset limpio en formato Arrow:

```py
drug_dataset_clean.save_to_disk("drug-reviews")
```

Esto creará una carpeta con la siguiente estructura:

```
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
```

en las que podemos ver que cada parte del dataset está asociada con una tabla *dataset.arrow* y algunos metadatos en *dataset_info.json* y *state.json*. Puedes pensar en el formato Arrow como una tabla sofisticada de columnas y filas que está optimizada para construir aplicaciones de alto rendimiento que procesan y transportan datasets grandes.

Una vez el dataset está guardado, podemos cargarlo usando la función `load_from_disk()` así:

```py
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
drug_dataset_reloaded
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
```

Para los formatos CSV y JSON, tenemos que guardar cada parte en un archivo separado. Una forma de hacerlo es iterando sobre las llaves y valores del objeto `DatasetDict`:

```py
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
```

Esto guarda cada parte en formato [JSON Lines](https://jsonlines.org), donde cada fila del dataset está almacenada como una única línea de JSON. Así se ve el primer ejemplo:

```py
!head -n 1 drug-reviews-train.jsonl
```

```python out
{"patient_id":141780,"drugName":"Escitalopram","condition":"depression","review":"\"I seemed to experience the regular side effects of LEXAPRO, insomnia, low sex drive, sleepiness during the day. I am taking it at night because my doctor said if it made me tired to take it at night. I assumed it would and started out taking it at night. Strange dreams, some pleasant. I was diagnosed with fibromyalgia. Seems to be helping with the pain. Have had anxiety and depression in my family, and have tried quite a few other medications that haven't worked. Only have been on it for two weeks but feel more positive in my mind, want to accomplish more in my life. Hopefully the side effects will dwindle away, worth it to stick with it from hearing others responses. Great medication.\"","rating":9.0,"date":"May 29, 2011","usefulCount":10,"review_length":125}
```

Podemos usar las técnicas de la [sección 2](/course/chapter5/2) para cargar los archivos JSON de la siguiente manera:

```py
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
```

Esto es todo lo que vamos a ver en nuestra revisión del manejo de datos con 🤗 Datasets. Ahora que tenemos un dataset limpio para entrenar un modelo, aquí van algunas ideas que podrías intentar:

1. Usa las técnicas del [Capítulo 3](/course/chapter3) para entrenar un clasificador que pueda predecir la condición del paciente con base en las reseñas de los medicamentos.
2. Usa el pipeline de `summarization` del [Capítulo 1](/course/chapter1) para generar resúmenes de las reseñas.

En la siguiente sección veremos cómo 🤗 Datasets te puede ayudar a trabajar con datasets enormes ¡sin explotar tu computador!


---

# ¿Big data? 🤗 ¡Datasets al rescate!


Hoy en día es común que tengas que trabajar con dataset de varios GB, especialmente si planeas pre-entrenar un transformador como BERT o GPT-2 desde ceros. En estos casos, _solamente cargar_ los datos puede ser un desafío. Por ejemplo, el corpus de WebText utilizado para preentrenar GPT-2 consiste de más de 8 millones de documentos y 40 GB de texto. ¡Cargarlo en la RAM de tu computador portátil le va a causar un paro cardíaco!

Afortunadamente, 🤗 Datasets está diseñado para superar estas limitaciones: te libera de problemas de manejo de memoria al tratar los datasets como archivos _proyectados en memoria_ (_memory-mapped_) y de límites de almacenamiento al hacer _streaming_ de las entradas en un corpus.


**Video:** [Ver en YouTube](https://youtu.be/JwISwTCPPWo)


En esta sección vamos a explorar estas funcionalidades de 🤗 Datasets con un corpus enorme de 825 GB conocido como el [Pile](https://pile.eleuther.ai). ¡Comencemos!

## ¿Qué es el Pile?

El _Pile_ es un corpus de textos en inglés creado por [EleutherAI](https://www.eleuther.ai) para entrenar modelos de lenguaje de gran escala. Incluye una selección diversa de datasets que abarca artículos científicos, repositorios de código de Github y texto filtrado de la web. El corpus de entrenamiento está disponible en [partes de 14 GB](https://mystic.the-eye.eu/public/AI/pile/) y también puedes descargar varios de los [componentes individuales](https://mystic.the-eye.eu/public/AI/pile_preliminary_components/). Arranquemos viendo el dataset de los abstracts de PubMed, un corpus de abstracts de 15 millones de publicaciones biomédicas en [PubMed](https://pubmed.ncbi.nlm.nih.gov/). Este dataset está en formato [JSON Lines](https://jsonlines.org) y está comprimido con la librería `zstandard`, así que primero tenemos que instalarla:

```py
!pip install zstandard
```

A continuación, podemos cargar el dataset usando el método para archivos remotos que aprendimos en la [sección 2](/course/chapter5/2):

```py
from datasets import load_dataset

# Esto toma algunos minutos para ejecutarse, así que ve por un te o un café mientras esperas :)
data_files = "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
pubmed_dataset
```

```python out
Dataset({
    features: ['meta', 'text'],
    num_rows: 15518009
})
```

Como podemos ver, hay 15.518.009 filas y dos columnas en el dataset, ¡un montón!

> [!TIP]
> ✎ Por defecto, 🤗 Datasets va a descomprimir los archivos necesarios para cargar un dataset. Si quieres ahorrar espacio de almacenamiento, puedes usar `DownloadConfig(delete_extracted=True)` al argumento `download_config` de `load_dataset()`. Revisa la [documentación](https://huggingface.co/docs/datasets/package_reference/builder_classes#datasets.DownloadConfig) para más detalles.

Veamos el contenido del primer ejemplo:

```py
pubmed_dataset[0]
```

```python out
{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
```

Ok, esto parece el abstract de un artículo médico. Ahora miremos cuánta RAM hemos usado para cargar el dataset.

## La magia de la proyección en memoria

Una forma simple de medir el uso de memoria en Python es con la librería [`psutil`](https://psutil.readthedocs.io/en/latest/), que se puede instalar con `pip` así:

```python
!pip install psutil
```

Esta librería contiene una clase `Process` que nos permite revisar el uso de memoria del proceso actual:

```py
import psutil

# Process.memory_info está expresado en bytes, así que lo convertimos en megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
```

```python out
RAM used: 5678.33 MB
```

El atributo `rss` se refiere al _resident set size_, que es la fracción de memoria que un proceso ocupa en RAM. Esta medición también incluye la memoria usada por el intérprete de Python y las librerías que hemos cargado, así que la cantidad real de memoria usada para cargar el dataset es un poco más pequeña. A modo de comparación, veamos qué tan grande es el dataset en disco, usando el atributo `dataset_size`. Dado que el resultado está expresado en bytes, tenemos que convertirlo manualmente en gigabytes:

```py
print(f"Number of files in dataset : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
```

```python out
Number of files in dataset : 20979437051
Dataset size (cache file) : 19.54 GB
```

Bien, a pesar de que el archivo es de casi 20 GB, ¡podemos cargarlo y acceder a su contenido con mucha menos RAM!

> [!TIP]
> ✏️ **¡Inténtalo!** Escoge alguno de los [subconjuntos](https://mystic.the-eye.eu/public/AI/pile_preliminary_components/) del _Pile_ que sea más grande que la RAM de tu computador portátil o de escritorio, cárgalo con 🤗 Datasets y mide la cantidad de RAM utilizada. Recuerda que para tener una medición precisa, tienes que hacerlo en un nuevo proceso. Puedes encontrar los tamaños de cada uno de los subconjuntos sin comprimir en la Tabla 1 del [paper de _Pile_](https://arxiv.org/abs/2101.00027).

Si estás familiarizado con Pandas, este resultado puede ser sorprendente por la famosa [regla de Wes Kinney](https://wesmckinney.com/blog/apache-arrow-pandas-internals/) que indica que típicamente necesitas de 5 a 10 veces la RAM que el tamaño del archivo de tu dataset. ¿Cómo resuelve entonces 🤗 Datasets este problema de manejo de memoria? 🤗 Datasets trata cada dataset como un [archivo proyectado en memoria](https://en.wikipedia.org/wiki/Memory-mapped_file), lo que permite un mapeo entre la RAM y el sistema de almacenamiento de archivos, que le permite a la librería acceder y operar los elementos del dataset sin necesidad de tenerlos cargados completamente en memoria.

Los archivos proyectados en memoria también pueden ser compartidos por múltiples procesos, lo que habilita la paralelización de métodos como `Dataset.map()` sin que sea obligatorio mover o copiar el dataset. Internamente, estas capacidades se logran gracias al formato de memoria [Apache Arrow](https://arrow.apache.org) y la librería [`pyarrow`](https://arrow.apache.org/docs/python/index.html), que permiten la carga y procesamiento de datos a gran velocidad. (Para ahondar más en Apache Arrow y algunas comparaciones con Pandas, revisa el [blog de Dejan Simic](https://towardsdatascience.com/apache-arrow-read-dataframe-with-zero-memory-69634092b1a)). Para verlo en acción, ejecutemos un test de velocidad iterando sobre todos los elementos del dataset de abstracts de PubMed:

```py
import timeit

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)
```

```python out
'Iterated over 15518009 examples (about 19.5 GB) in 64.2s, i.e. 0.304 GB/s'
```

Aquí usamos el módulo `timeit` de Python para medir el tiempo de ejecución que se toma `code_snippet`. Típicamemente, puedes iterar a lo largo de un dataset a una velocidad de unas cuantas décimas de un GB por segundo. Esto funciona muy bien para la gran mayoría de aplicaciones, pero algunas veces tendrás que trabajar con un dataset que es tan grande para incluso almacenarse en el disco de tu computador. Por ejemplo, si quisieramos descargar el _Pile_ completo ¡necesitaríamos 825 GB de almacenamiento libre! Para trabajar con esos casos, 🤗 Datasets puede trabajar haciendo _streaming_, lo que permite la descarga y acceso a los elementos sobre la marcha, sin necesidad de descargar todo el dataset. Veamos cómo funciona:

> [!TIP]
> 💡 En los cuadernos de Jupyter también puedes medir el tiempo de ejecución de las celdas usando [`%%timeit`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit).

## Haciendo _streaming_ de datasets

Para habilitar el _streaming_ basta con pasar el argumento `streaming=True` a la función `load_dataset()`. Por ejemplo, carguemos el dataset de abstracts de PubMed de nuevo, pero en modo _streaming_.

```py
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
```

En vez del `Dataset` común y corriente que nos hemos encontrado en el resto del capítulo, el objeto devuelto con `streaming=True` es un `IterableDataset`. Como su nombre lo indica, para acceder a los elementos de un `IterableDataset` tenemos que iterar sobre él. Podemos acceder al primer elemento de nuestro dataset de la siguiente manera:

```py
next(iter(pubmed_dataset_streamed))
```

```python out
{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
```

Los elementos de un dataset _streamed_ pueden ser procesados sobre la marcha usando `IterableDataset.map()`, lo que puede servirte si tienes que tokenizar los inputs. El proceso es exactamente el mismo que el que usamos para tokenizar nuestro dataset en el [Capítulo 3](/course/chapter3), con la única diferencia de que los outputs se devuelven uno por uno.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
next(iter(tokenized_dataset))
```

```python out
{'input_ids': [101, 4958, 5178, 4328, 6779, ...], 'attention_mask': [1, 1, 1, 1, 1, ...]}
```

> [!TIP]
> 💡 Para acelerar la tokenización con _streaming_ puedes definir `batched=True`, como lo vimos en la sección anterior. Esto va a procesar los ejemplos lote por lote. Recuerda que el tamaño por defecto de los lotes es 1.000 y puede ser especificado con el argumento `batch_size`.

También puedes aleatorizar el orden de un dataset _streamed_ usando `IterableDataset.shuffle()`, pero a diferencia de `Dataset.shuffle()` esto sólo afecta a los elementos en un `buffer_size` determinado:

```py
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))
```

```python out
{'meta': {'pmid': 11410799, 'language': 'eng'},
 'text': 'Randomized study of dose or schedule modification of granulocyte colony-stimulating factor in platinum-based chemotherapy for elderly patients with lung cancer ...'}
```

En este ejemplo, seleccionamos un ejemplo aleatorio de los primeros 10.000 ejemplos en el buffer. Apenas se accede a un ejemplo, su lugar en el buffer se llena con el siguiente ejemplo en el corpus (i.e., el ejemplo número 10.001). También puedes seleccionar elementos de un dataset _streamed_ usando las funciones `IterableDataset.take()` y `IterableDataset.skip()`, que funcionan de manera similar a `Dataset.select()`. Por ejemplo, para seleccionar los 5 primeros ejemplos en el dataset de abstracts de PubMed podemos hacer lo siguiente:

```py
dataset_head = pubmed_dataset_streamed.take(5)
list(dataset_head)
```

```python out
[{'meta': {'pmid': 11409574, 'language': 'eng'},
  'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection ...'},
 {'meta': {'pmid': 11409575, 'language': 'eng'},
  'text': 'Clinical signs of hypoxaemia in children with acute lower respiratory infection: indicators of oxygen therapy ...'},
 {'meta': {'pmid': 11409576, 'language': 'eng'},
  'text': "Hypoxaemia in children with severe pneumonia in Papua New Guinea ..."},
 {'meta': {'pmid': 11409577, 'language': 'eng'},
  'text': 'Oxygen concentrators and cylinders ...'},
 {'meta': {'pmid': 11409578, 'language': 'eng'},
  'text': 'Oxygen supply in rural africa: a personal experience ...'}]
```

También podemos usar la función `IterableDataset.skip()` para crear conjuntos de entrenamiento y validación de un dataset ordenado aleatoriamente así:

```py
# Salta las primeras 1000 muestras e incluye el resto en el conjunto de entrenamiento
train_dataset = shuffled_dataset.skip(1000)
# Toma las primeras 1000 muestras para el conjunto de validación
validation_dataset = shuffled_dataset.take(1000)
```

Vamos a repasar la exploración del _streaming_ de datasets con una aplicación común: combinar múltiples datasets para crear un solo corpus. 🤗 Datasets provee una función `interleave_datasets()` que convierte una lista de objetos `IterableDataset` en un solo `IterableDataset`, donde la lista de elementos del nuevo dataset se obtiene al alternar entre los ejemplos originales. Esta función es particularmente útil cuando quieres combinar datasets grandes, así que como ejemplo hagamos _streaming_ del conjunto FreeLaw del _Pile_, que es un dataset de 51 GB con opiniones legales de las cortes en Estados Unidos.

```py
law_dataset_streamed = load_dataset(
    "json",
    data_files="https://mystic.the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
next(iter(law_dataset_streamed))
```

```python out
{'meta': {'case_ID': '110921.json',
  'case_jurisdiction': 'scotus.tar.gz',
  'date_created': '2010-04-28T17:12:49Z'},
 'text': '\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...'}
```

Este dataset es lo suficientemente grande como para llevar al límite la RAM de la mayoría de computadores portátiles. Sin embargo, ¡podemos cargarla y acceder a el sin esfuerzo! Ahora combinemos los ejemplos de FreeLaw y PubMed usando la función `interleave_datasets()`:

```py
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))
```

```python out
[{'meta': {'pmid': 11409574, 'language': 'eng'},
  'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection ...'},
 {'meta': {'case_ID': '110921.json',
   'case_jurisdiction': 'scotus.tar.gz',
   'date_created': '2010-04-28T17:12:49Z'},
  'text': '\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...'}]
```

Usamos la función `islice()` del módulo `itertools` de Python para seleccionar los primeros dos ejemplos del dataset combinado y podemos ver que corresponden con los primeros dos ejemplos de cada uno de los dos datasets de origen.

Finalmente, si quieres hacer _streaming_ del _Pile_ de 825 GB en su totalidad, puedes usar todos los archivos preparados de la siguiente manera:

```py
base_url = "https://mystic.the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
next(iter(pile_dataset["train"]))
```

```python out
{'meta': {'pile_set_name': 'Pile-CC'},
 'text': 'It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web...'}
```

> [!TIP]
> ✏️ **¡Inténtalo!** Usa alguno de los corpus grandes de Common Crawl como [`mc4`](https://huggingface.co/datasets/mc4) u [`oscar`](https://huggingface.co/datasets/oscar) para crear un dataset _streaming_ multilenguaje que represente las proporciones de lenguajes hablados en un país de tu elección. Por ejemplo, los 4 lenguajes nacionales en Suiza son alemán, francés, italiano y romanche, así que podrías crear un corpus suizo al hacer un muestreo de Oscar de acuerdo con su proporción de lenguaje.

Ya tienes todas las herramientas para cargar y procesar datasets de todas las formas y tamaños, pero a menos que seas muy afortunado, llegará un punto en tu camino de PLN en el que tendrás que crear el dataset tu mismo para resolver tu problema particular. De esto hablaremos en la siguiente sección.


---

# Crea tu propio dataset


Algunas veces el dataset que necesitas para crear una aplicación de procesamiento de lenguaje natural no existe, así que necesitas crearla. En esta sección vamos a mostrarte cómo crear un corpus de [issues de GitHub](https://github.com/features/issues/), que se usan comúnmente para rastrear bugs o features en repositorios de GitHub. Este corpus podría ser usado para varios propósitos como:

* Explorar qué tanto se demora el cierre un issue abierto o un pull request
* Entrenar un _clasificador de etiquetas múltiples_ que pueda etiquetar issues con metadados basado en la descripción del issue (e.g., "bug", "mejora" o "pregunta")
* Crear un motor de búsqueda semántica para encontrar qué issues coinciden con la pregunta del usuario

En esta sección nos vamos a enfocar en la creación del corpus y en la siguiente vamos a abordar la aplicación de búsqueda semántica. Para que esto sea un meta-proyecto, vamos a usar los issues de GitHub asociados con un proyecto popular de código abierto: 🤗 Datasets! Veamos cómo obtener los datos y explorar la información contenida en estos issues.

## Obteniendo los datos

Puedes encontrar todos los issues de 🤗 Datasets yendo a la [pestaña de issues](https://github.com/huggingface/datasets/issues) del repositorio. Como se puede ver en la siguiente captura de pantalla, al momento de escribir esta sección habían 331 issues abiertos y 668 cerrados.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues.png" alt="The GitHub issues associated with 🤗 Datasets." width="80%"/>
</div>

Si haces clic en alguno de estos issues te encontrarás con que incluyen un título, una descripción y un conjunto de etiquetas que lo caracterizan. Un ejemplo de esto se muestra en la siguiente captura de pantalla.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues-single.png" alt="A typical GitHub issue in the 🤗 Datasets repository." width="80%"/>
</div>

Para descargar todos los issues del repositorio, usaremos el [API REST de GitHub](https://docs.github.com/en/rest) para obtener el [endpoint `Issues`](https://docs.github.com/en/rest/reference/issues#list-repository-issues). Este endpoint devuelve una lista de objetos JSON, en la que cada objeto contiene un gran número de campos que incluyen el título y la descripción, así como metadatos sobre el estado del issue, entre otros.

Una forma conveniente de descargar los issues es a través de la librería `requests`, que es la manera estándar para hacer pedidos HTTP en Python. Puedes instalar esta librería instalando:

```python
!pip install requests
```

Una vez la librería está instalada, puedes hacer pedidos GET al endpoint `Issues` ejecutando la función `requests.get()`. Por ejemplo, puedes correr el siguiente comando para obtener el primer issue de la primera página:

```py
import requests

url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
```

El objeto `response` contiene una gran cantidad de información útil sobre el pedido, incluyendo el código de status de HTTP:

```py
response.status_code
```

```python out
200
```

en el que un código de `200` significa que el pedido fue exitoso (puedes ver una lista de posibles códigos de status de HTTP [aquí](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)). No obstante, en lo que estamos interesados realmente es el _payload_, que se puede acceder en varios formatos como bytes, strings o JSON. Como ya sabemos que los issues están en formato JSON, inspeccionemos el _payload_ de la siguiente manera:

```py
response.json()
```

```python out
[{'url': 'https://api.github.com/repos/huggingface/datasets/issues/2792',
  'repository_url': 'https://api.github.com/repos/huggingface/datasets',
  'labels_url': 'https://api.github.com/repos/huggingface/datasets/issues/2792/labels{/name}',
  'comments_url': 'https://api.github.com/repos/huggingface/datasets/issues/2792/comments',
  'events_url': 'https://api.github.com/repos/huggingface/datasets/issues/2792/events',
  'html_url': 'https://github.com/huggingface/datasets/pull/2792',
  'id': 968650274,
  'node_id': 'MDExOlB1bGxSZXF1ZXN0NzEwNzUyMjc0',
  'number': 2792,
  'title': 'Update GooAQ',
  'user': {'login': 'bhavitvyamalik',
   'id': 19718818,
   'node_id': 'MDQ6VXNlcjE5NzE4ODE4',
   'avatar_url': 'https://avatars.githubusercontent.com/u/19718818?v=4',
   'gravatar_id': '',
   'url': 'https://api.github.com/users/bhavitvyamalik',
   'html_url': 'https://github.com/bhavitvyamalik',
   'followers_url': 'https://api.github.com/users/bhavitvyamalik/followers',
   'following_url': 'https://api.github.com/users/bhavitvyamalik/following{/other_user}',
   'gists_url': 'https://api.github.com/users/bhavitvyamalik/gists{/gist_id}',
   'starred_url': 'https://api.github.com/users/bhavitvyamalik/starred{/owner}{/repo}',
   'subscriptions_url': 'https://api.github.com/users/bhavitvyamalik/subscriptions',
   'organizations_url': 'https://api.github.com/users/bhavitvyamalik/orgs',
   'repos_url': 'https://api.github.com/users/bhavitvyamalik/repos',
   'events_url': 'https://api.github.com/users/bhavitvyamalik/events{/privacy}',
   'received_events_url': 'https://api.github.com/users/bhavitvyamalik/received_events',
   'type': 'User',
   'site_admin': False},
  'labels': [],
  'state': 'open',
  'locked': False,
  'assignee': None,
  'assignees': [],
  'milestone': None,
  'comments': 1,
  'created_at': '2021-08-12T11:40:18Z',
  'updated_at': '2021-08-12T12:31:17Z',
  'closed_at': None,
  'author_association': 'CONTRIBUTOR',
  'active_lock_reason': None,
  'pull_request': {'url': 'https://api.github.com/repos/huggingface/datasets/pulls/2792',
   'html_url': 'https://github.com/huggingface/datasets/pull/2792',
   'diff_url': 'https://github.com/huggingface/datasets/pull/2792.diff',
   'patch_url': 'https://github.com/huggingface/datasets/pull/2792.patch'},
  'body': '[GooAQ](https://github.com/allenai/gooaq) dataset was recently updated after splits were added for the same. This PR contains new updated GooAQ with train/val/test splits and updated README as well.',
  'performed_via_github_app': None}]
```

Wow, ¡es mucha información! Podemos ver campos útiles como `title`, `body` y `number`, que describen el issue, así como información del usuario de GitHub que lo abrió.

> [!TIP]
> ✏️ **¡Inténtalo!** Haz clic en algunas de las URL en el _payload_ JSON de arriba para explorar la información que está enlazada al issue de GitHub.

Tal como se describe en la [documentación](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting) de GitHub, los pedidos sin autenticación están limitados a 60 por hora. Si bien puedes incrementar el parámetro de búsqueda `per_page` para reducir el número de pedidos que haces, igual puedes alcanzar el límite de pedidos en cualquier repositorio que tenga más que un par de miles de issues. En vez de hacer eso, puedes seguir las [instrucciones](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) de GitHub para crear un _token de acceso personal_ y que puedas incrementar el límite de pedidos a 5.000 por hora. Una vez tengas tu token, puedes incluirlo como parte del encabezado del pedido:

```py
GITHUB_TOKEN = xxx  # Copy your GitHub token here
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
```

> [!WARNING]
> ⚠️ No compartas un cuaderno que contenga tu `GITHUB_TOKEN`. Te recomendamos eliminar la última celda una vez la has ejecutado para evitar filtrar accidentalmente esta información. Aún mejor, guarda el token en un archivo *.env* y usa la librería [`python-dotenv`](https://github.com/theskumar/python-dotenv) para cargarla automáticamente como una variable de ambiente.

Ahora que tenemos nuestro token de acceso, creemos una función que descargue todos los issues de un repositorio de GitHub:

```py
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Número de issues por página
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query con state=all para obtener tanto issues abiertos como cerrados
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Vacía el batch para el siguiente periodo de tiempo
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )
```

Cuando ejecutemos `fetch_issues()`, se descargarán todos los issues en lotes para evitar exceder el límite de GitHub sobre el número de pedidos por hora. El resultado se guardará en un archivo _repository_name-issues.jsonl_, donde cada línea es un objeto JSON que representa un issue. Usemos esta función para cargar todos los issues de 🤗 Datasets:

```py
# Dependiendo de tu conexión a internet, esto puede tomar varios minutos para ejecutarse...
fetch_issues()
```

Una vez los issues estén descargados, los podemos cargar localmente usando las habilidades aprendidas en la [sección 2](/course/chapter5/2):

```py
issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
issues_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'timeline_url', 'performed_via_github_app'],
    num_rows: 3019
})
```

¡Genial! Hemos creado nuestro primer dataset desde cero. Pero, ¿por qué hay varios miles de issues cuando la [pestaña de Issues](https://github.com/huggingface/datasets/issues) del repositorio de 🤗 Datasets sólo muestra alrededor de 1.000 en total? Como se describe en la [documentación](https://docs.github.com/en/rest/reference/issues#list-issues-assigned-to-the-authenticated-user), esto sucede porque también descargamos todos los pull requests:

> GitHub's REST API v3 considers every pull request an issue, but not every issue is a pull request. For this reason, "Issues" endpoints may return both issues and pull requests in the response. You can identify pull requests by the `pull_request` key. Be aware that the `id` of a pull request returned from "Issues" endpoints will be an issue id.

Como el contenido de los issues y pull requests son diferentes, hagamos un preprocesamiento simple para distinguirlos entre sí.

## Limpiando los datos

El fragmento anterior de la documentación de GitHub nos dice que la columna `pull_request` puede usarse para diferenciar los issues de los pull requests. Veamos una muestra aleatoria para ver la diferencia. Como hicimos en la [sección 3](/course/chapter5/3), vamos a encadenar `Dataset.shuffle()` y `Dataset.select()` para crear una muestra aleatoria y luego unir las columnas de `html_url` y `pull_request` para comparar las distintas URL:

```py
sample = issues_dataset.shuffle(seed=666).select(range(3))

# Imprime la URL y las entradas de pull_request
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")
```

```python out
>> URL: https://github.com/huggingface/datasets/pull/850
>> Pull request: {'url': 'https://api.github.com/repos/huggingface/datasets/pulls/850', 'html_url': 'https://github.com/huggingface/datasets/pull/850', 'diff_url': 'https://github.com/huggingface/datasets/pull/850.diff', 'patch_url': 'https://github.com/huggingface/datasets/pull/850.patch'}

>> URL: https://github.com/huggingface/datasets/issues/2773
>> Pull request: None

>> URL: https://github.com/huggingface/datasets/pull/783
>> Pull request: {'url': 'https://api.github.com/repos/huggingface/datasets/pulls/783', 'html_url': 'https://github.com/huggingface/datasets/pull/783', 'diff_url': 'https://github.com/huggingface/datasets/pull/783.diff', 'patch_url': 'https://github.com/huggingface/datasets/pull/783.patch'}
```

Podemos ver que cada pull request está asociado con varias URL, mientras que los issues ordinarios tienen una entrada `None`. Podemos usar esta distinción para crear una nueva columna `is_pull_request` que revisa si el campo `pull_request` es `None` o no:

```py
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)
```

> [!TIP]
> ✏️ **¡Inténtalo!** Calcula el tiempo promedio que toma cerrar issues en 🤗 Datasets. La función `Dataset.filter()` te será útil para filtrar los pull requests y los issues abiertos, y puedes usar la función `Dataset.set_format()` para convertir el dataset a un `DataFrame` para poder manipular fácilmente los timestamps de `created_at` y `closed_at`. Para puntos extra, calcula el tiempo promedio que toma cerrar pull requests.

Si bien podemos limpiar aún más el dataset eliminando o renombrando algunas columnas, es una buena práctica mantener un dataset lo más parecido al original en esta etapa, para que se pueda usar fácilmente en varias aplicaciones.

Antes de subir el dataset el Hub de Hugging Face, nos hace falta añadirle algo más: los comentarios asociados con cada issue y pull request. Los vamos a añadir con el API REST de GitHub.

## Ampliando el dataset

Como se muestra en la siguiente captura de pantalla, los comentarios asociados con un issue o un pull request son una fuente rica de información, especialmente si estamos interesados en construir un motor de búsqueda para responder preguntas de usuarios sobre la librería.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues-comment.png" alt="Comments associated with an issue about 🤗 Datasets." width="80%"/>
</div>

El API REST de GitHub tiene un [endpoint `Comments`](https://docs.github.com/en/rest/reference/issues#list-issue-comments) que devuelve todos los comentarios asociados con un número de issue. Probemos este endpoint para ver qué devuelve:

```py
issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
response.json()
```

```python out
[{'url': 'https://api.github.com/repos/huggingface/datasets/issues/comments/897594128',
  'html_url': 'https://github.com/huggingface/datasets/pull/2792#issuecomment-897594128',
  'issue_url': 'https://api.github.com/repos/huggingface/datasets/issues/2792',
  'id': 897594128,
  'node_id': 'IC_kwDODunzps41gDMQ',
  'user': {'login': 'bhavitvyamalik',
   'id': 19718818,
   'node_id': 'MDQ6VXNlcjE5NzE4ODE4',
   'avatar_url': 'https://avatars.githubusercontent.com/u/19718818?v=4',
   'gravatar_id': '',
   'url': 'https://api.github.com/users/bhavitvyamalik',
   'html_url': 'https://github.com/bhavitvyamalik',
   'followers_url': 'https://api.github.com/users/bhavitvyamalik/followers',
   'following_url': 'https://api.github.com/users/bhavitvyamalik/following{/other_user}',
   'gists_url': 'https://api.github.com/users/bhavitvyamalik/gists{/gist_id}',
   'starred_url': 'https://api.github.com/users/bhavitvyamalik/starred{/owner}{/repo}',
   'subscriptions_url': 'https://api.github.com/users/bhavitvyamalik/subscriptions',
   'organizations_url': 'https://api.github.com/users/bhavitvyamalik/orgs',
   'repos_url': 'https://api.github.com/users/bhavitvyamalik/repos',
   'events_url': 'https://api.github.com/users/bhavitvyamalik/events{/privacy}',
   'received_events_url': 'https://api.github.com/users/bhavitvyamalik/received_events',
   'type': 'User',
   'site_admin': False},
  'created_at': '2021-08-12T12:21:52Z',
  'updated_at': '2021-08-12T12:31:17Z',
  'author_association': 'CONTRIBUTOR',
  'body': "@albertvillanova my tests are failing here:\r\n```\r\ndataset_name = 'gooaq'\r\n\r\n    def test_load_dataset(self, dataset_name):\r\n        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]\r\n>       self.dataset_tester.check_load_dataset(dataset_name, configs, is_local=True, use_local_dummy_data=True)\r\n\r\ntests/test_dataset_common.py:234: \r\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \r\ntests/test_dataset_common.py:187: in check_load_dataset\r\n    self.parent.assertTrue(len(dataset[split]) > 0)\r\nE   AssertionError: False is not true\r\n```\r\nWhen I try loading dataset on local machine it works fine. Any suggestions on how can I avoid this error?",
  'performed_via_github_app': None}]
```

Podemos ver que el comentario está almacenado en el campo `body`, así que escribamos una función simple que devuelva todos los comentarios asociados con un issue al extraer el contenido de `body` para cada elemento en el `response.json()`:

```py
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


# Revisar que el comportamiento de nuestra función es el esperado
get_comments(2792)
```

```python out
["@albertvillanova my tests are failing here:\r\n```\r\ndataset_name = 'gooaq'\r\n\r\n    def test_load_dataset(self, dataset_name):\r\n        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]\r\n>       self.dataset_tester.check_load_dataset(dataset_name, configs, is_local=True, use_local_dummy_data=True)\r\n\r\ntests/test_dataset_common.py:234: \r\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \r\ntests/test_dataset_common.py:187: in check_load_dataset\r\n    self.parent.assertTrue(len(dataset[split]) > 0)\r\nE   AssertionError: False is not true\r\n```\r\nWhen I try loading dataset on local machine it works fine. Any suggestions on how can I avoid this error?"]
```

Esto luce bien, así que usemos `Dataset.map()` para añadir una nueva columna `comments` a cada issue en el dataset:

```py
# Dependiendo de tu conexión a internet, esto puede tomar varios minutos...
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)
```

El último paso es guardar el dataset ampliado en el mismo lugar que los datos originales para poderlos subir al Hub:

```py
issues_with_comments_dataset.to_json("issues-datasets-with-comments.jsonl")
```

## Subiendo un dataset al Hub de Hugging Face


**Video:** [Ver en YouTube](https://youtu.be/HaN6qCr_Afc)


Ahora que tenemos nuestro dataset ampliado, es momento de subirlo al Hub para poder compartirlo con la comunidad. Para subir el dataset tenemos que usar la [librería 🤗 Hub](https://github.com/huggingface/huggingface_hub), que nos permite interactuar con el Hub de Hugging Face usando una API de Python. 🤗 Hub viene instalada con 🤗 Transformers, así que podemos usarla directamente. Por ejemplo, podemos usar la función `list_datasets()` para obtener información sobre todos los datasets públicos que están almacenados en el Hub:

```py
from huggingface_hub import list_datasets

all_datasets = list_datasets()
print(f"Number of datasets on Hub: {len(all_datasets)}")
print(all_datasets[0])
```

```python out
Number of datasets on Hub: 1487
Dataset Name: acronym_identification, Tags: ['annotations_creators:expert-generated', 'language_creators:found', 'languages:en', 'licenses:mit', 'multilinguality:monolingual', 'size_categories:10K<n<100K', 'source_datasets:original', 'task_categories:structure-prediction', 'task_ids:structure-prediction-other-acronym-identification']
```
Podemos ver que hay alrededor de 1.500 datasets en el Hub y que la función `list_datasets()` también provee algunos metadatos sobre el repositorio de cada uno.

Para lo que queremos hacer, lo primero que necesitamos es crear un nuevo repositorio de dataset en el Hub. Para ello, necesitamos un token de autenticación, que se obtiene al acceder al Hub de Hugging Face con la función `notebook_login()`:

```py
from huggingface_hub import notebook_login

notebook_login()
```

Esto crea un widget en el que ingresas tu nombre de usuario y contraseña, y guarda un token API en *~/.huggingface/token*. Si estás ejecutando el código en la terminal, puedes acceder a través de la línea de comandos así:

```bash
huggingface-cli login
```

Una vez hecho esto, podemos crear un nuevo repositorio para el dataset con la función `create_repo()`:

```py
from huggingface_hub import create_repo

repo_url = create_repo(name="github-issues", repo_type="dataset")
repo_url
```

```python out
'https://huggingface.co/datasets/lewtun/github-issues'
```

En este ejemplo, hemos creado un repositorio vacío para el dataset llamado `github-issues` bajo el nombre de usuario `lewtun` (¡el nombre de usuario debería ser tu nombre de usuario del Hub cuando estés ejecutando este código!).

> [!TIP]
> ✏️ **¡Inténtalo!** Usa tu nombre de usuario de Hugging Face Hub para obtener un token y crear un repositorio vacío llamado `github-issues`. Recuerda **nunca guardar tus credenciales** en Colab o cualquier otro repositorio, ya que esta información puede ser aprovechada por terceros.

Ahora clonemos el repositorio del Hub a nuestra máquina local y copiemos nuestro dataset ahí. 🤗 Hub incluye una clase `Repositorio` que envuelve muchos de los comandos comunes de Git, así que para clonar el repositorio remoto solamente necesitamos dar la URL y la ruta local en la que lo queremos clonar:

```py
from huggingface_hub import Repository

repo = Repository(local_dir="github-issues", clone_from=repo_url)
!cp issues-datasets-with-comments.jsonl github-issues/
```

Por defecto, varias extensiones de archivo (como *.bin*, *.gz*, and *.zip*) se siguen con Git LFS de tal manera que los archivos grandes se pueden versionar dentro del mismo flujo de trabajo de Git. Puedes encontrar una lista de extensiones que se van a seguir en el archivo *.gitattributes*. Para incluir el formato JSON Lines en la lista, puedes ejecutar el siguiente comando:

```py
repo.lfs_track("*.jsonl")
```

Luego, podemos usar `$$Repository.push_to_hub()` para subir el dataset al Hub:

```py
repo.push_to_hub()
```

Si navegamos a la URL que aparece en `repo_url`, deberíamos ver que el archivo del dataset se ha subido.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/hub-repo.png" alt="Our dataset repository on the Hugging Face Hub." width="80%"/>
</div>

Desde aqui, cualquier persona podrá descargar el dataset incluyendo el ID del repositorio en el argumento `path` de la función `load_dataset()`:

```py
remote_dataset = load_dataset("lewtun/github-issues", split="train")
remote_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 2855
})
```

¡Genial, hemos subido el dataset al Hub y ya está disponible para que otras personas lo usen! Sólo hay una cosa restante por hacer: añadir una _tarjeta del dataset_ (_dataset card_) que explique cómo se creó el corpus y provea información útil para la comunidad.

> [!TIP]
> 💡 También puedes subir un dataset al Hub de Hugging Face directamente desde la terminal usando `huggingface-cli` y un poco de Git. Revisa la [guía de 🤗 Datasets](https://huggingface.co/docs/datasets/share#share-a-dataset-using-the-cli) para más detalles sobre cómo hacerlo.

## Creando una tarjeta del dataset

Los datasets bien documentados tienen más probabilidades de ser útiles para otros (incluyéndote a ti en el futuro), dado que brindan la información necesaria para que los usuarios decidan si el dataset es útil para su tarea, así como para evaluar cualquier sesgo o riesgo potencial asociado a su uso.

En el Hub de Hugging Face, esta información se almacena en el archivo *README.md* del repositorio del dataset. Hay dos pasos que deberías hacer antes de crear este archivo:

1. Usa la [aplicación `datasets-tagging`](https://huggingface.co/datasets/tagging/) para crear etiquetas de metadatos en el formato YAML. Estas etiquetas se usan para una variedad de funciones de búsqueda en el Hub de Hugging Face y aseguran que otros miembros de la comunidad puedan encontrar tu dataset. Dado que creamos un dataset personalizado en esta sección, tendremos que clonar el repositorio `datasets-tagging` y correr la aplicación localmente. Así se ve la interfaz de la aplicación:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-tagger.png" alt="The `datasets-tagging` interface." width="80%"/>
</div>

2. Lee la [guía de 🤗 Datasets](https://github.com/huggingface/datasets/blob/master/templates/README_guide.md) sobre cómo crear tarjetas informativas y usarlas como plantilla.

Puedes crear el archivo *README.md* directamente desde el Hub y puedes encontrar una plantilla de tarjeta en el repositorio `lewtun/github-issues`. Así se ve una tarjeta de dataset diligenciada:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/dataset-card.png" alt="A dataset card." width="80%"/>
</div>

> [!TIP]
> ✏️ **¡Inténtalo!** Usa la aplicación `dataset-tagging` y la [guía de 🤗 Datasets](https://github.com/huggingface/datasets/blob/master/templates/README_guide.md) para completar el archivo *README.md* para tu dataset de issues de GitHub.

¡Eso es todo! Hemos visto que crear un buen dataset requiere de mucho esfuerzo de tu parte, pero afortunadamente subirlo y compartirlo con la comunidad no. En la siguiente sección usaremos nuestro nuevo dataset para crear un motor de búsqueda semántica con 🤗 Datasets que pueda emparejar preguntas con los issues y comentarios más relevantes.

> [!TIP]
> ✏️ **¡Inténtalo!** Sigue los pasos descritos en esta sección para crear un dataset de issues de GitHub de tu librería de código abierto favorita (¡por supuesto, escoge algo distinto a 🤗 Datasets!). Para puntos extra, ajusta un clasificador de etiquetas múltiples para predecir las etiquetas presentes en el campo `labels`.


---



# Búsqueda semántica con FAISS


En la [sección 5](/course/chapter5/5) creamos un dataset de issues y comentarios del repositorio de GitHub de 🤗 Datasets. En esta sección usaremos esta información para construir un motor de búsqueda que nos ayude a responder nuestras preguntas más apremiantes sobre la librería.


**Video:** [Ver en YouTube](https://youtu.be/OATCgQtNX2o)


## Usando _embeddings_ para la búsqueda semántica

Como vimos en el [Capítulo 1](/course/chapter1), los modelos de lenguaje basados en Transformers representan cada token en un texto como un _vector de embeddings_. Resulta que podemos agrupar los _embeddings_ individuales en representaciones vectoriales para oraciones, párrafos o (en algunos casos) documentos completos. Estos _embeddings_ pueden ser usados para encontrar documentos similares en el corpus al calcular la similaridad del producto punto (o alguna otra métrica de similaridad) entre cada _embedding_ y devolver los documentos con la mayor coincidencia.

En esta sección vamos a usar _embeddings_ para desarrollar un motor de búsqueda semántica. Estos motores de búsqueda tienen varias ventajas sobre abordajes convencionales basados en la coincidencia de palabras clave en una búsqueda con los documentos.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search.svg" alt="Semantic search."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search-dark.svg" alt="Semantic search."/>
</div>

## Cargando y preparando el dataset

Lo primero que tenemos que hacer es descargar el dataset de issues de GitHub, así que usaremos la librería 🤗 Hub para resolver la URL en la que está almacenado nuestro archivo en el Hub de Hugging Face:

```py
from huggingface_hub import hf_hub_url

data_files = hf_hub_url(
    repo_id="lewtun/github-issues",
    filename="datasets-issues-with-comments.jsonl",
    repo_type="dataset",
)
```

Con la URL almacenada en `data_files`, podemos cargar el dataset remoto usando el método introducido en la [sección 2](/course/chapter5/2):

```py
from datasets import load_dataset

issues_dataset = load_dataset("json", data_files=data_files, split="train")
issues_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 2855
})
```

Hemos especificado el conjunto `train` por defecto en `load_dataset()`, de tal manera que devuelva un objeto `Dataset` en vez de un `DatasetDict`. Lo primero que debemos hacer es filtrar los pull requests, dado que estos no se suelen usar para resolver preguntas de usuarios e introducirán ruido en nuestro motor de búsqueda. Como ya debe ser familiar para ti, podemos usar la función `Dataset.filter()` para excluir estas filas en nuestro dataset. A su vez, filtremos las filas que no tienen comentarios, dado que no van a darnos respuestas para las preguntas de los usuarios.

```py
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
issues_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 771
})
```

Podemos ver que hay un gran número de columnas en nuestro dataset, muchas de las cuales no necesitamos para construir nuestro motor de búsqueda. Desde la perspectiva de la búsqueda, las columnas más informativas son `title`, `body` y `comments`, mientras que `html_url` nos indica un link al issue correspondiente. Usemos la función `Dataset.remove_columns()` para eliminar el resto:

```py
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 771
})
```

Para crear nuestros _embeddings_, vamos a ampliar cada comentario añadiéndole el título y el cuerpo del issue, dado que estos campos suelen incluir información de contexto útil. Dado que nuestra función `comments` es una lista de comentarios para cada issue, necesitamos "explotar" la columna para que cada fila sea una tupla `(html_url, title, body, comment)`. Podemos hacer esto en Pandas con la [función `DataFrame.explode()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html), que crea una nueva fila para cada elemento en una columna que está en forma de lista, al tiempo que replica el resto de los valores de las otras columnas. Para verlo en acción, primero debemos cambiar al formato `DataFrame` de Pandas:

```py
issues_dataset.set_format("pandas")
df = issues_dataset[:]
```

Si inspeccionamos la primera fila en este `DataFrame` podemos ver que hay 4 comentarios asociados con este issue:

```py
df["comments"][0].tolist()
```

```python out
['the bug code locate in ：\r\n    if data_args.task_name is not None:\r\n        # Downloading and loading a dataset from the hub.\r\n        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)',
 'Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com\r\n\r\nNormally, it should work if you wait a little and then retry.\r\n\r\nCould you please confirm if the problem persists?',
 'cannot connect，even by Web browser，please check that  there is some  problems。',
 'I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...']
```

Cuando "explotamos" `df`, queremos obtener una fila para cada uno de estos comentarios. Veamos si este es el caso:

```py
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)
```

<table border="1" class="dataframe" style="table-layout: fixed; word-wrap:break-word; width: 100%;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>html_url</th>
      <th>title</th>
      <th>comments</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>the bug code locate in ：\r\n    if data_args.task_name is not None...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>cannot connect，even by Web browser，please check that  there is some  problems。</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
  </tbody>
</table>

Genial, podemos ver que las filas se han replicado y que la columna `comments` incluye los comentarios individuales. Ahora que hemos terminado con Pandas, podemos volver a cambiar el formato a `Dataset` cargando el `DataFrame` en memoria: 

```py
from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 2842
})
```

¡Esto nos ha dado varios miles de comentarios con los que trabajar!

> [!TIP]
> ✏️ **¡Inténtalo!** Prueba si puedes usar la función `Dataset.map()` para "explotar" la columna `comments` en `issues_dataset` _sin_ necesidad de usar Pandas. Esto es un poco complejo; te recomendamos revisar la sección de ["Batch mapping"](https://huggingface.co/docs/datasets/about_map_batch#batch-mapping) de la documentación de 🤗 Datasets para completar esta tarea.

Ahora que tenemos un comentario para cada fila, creemos una columna `comments_length` que contenga el número de palabras por comentario:

```py
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
```

Podemos usar esta nueva columna para filtrar los comentarios cortos, que típicamente incluyen cosas como "cc @letwun" o "¡Gracias!", que no son relevantes para nuestro motor de búsqueda. No hay un número preciso que debamos filtrar, pero alrededor de 15 palabras es un buen comienzo:

```py
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length'],
    num_rows: 2098
})
```

Ahora que hemos limpiado un poco el dataset, vamos a concatenar el título, la descripción y los comentarios del issue en una nueva columna `text`. Como lo hemos venido haciendo, escribiremos una función para pasarla a `Dataset.map()`:

```py
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)
```

¡Por fin estamos listos para crear _embeddings_!

## Creando _embeddings_ de texto

En el [Capítulo 2](/course/chapter2) vimos que podemos obtener _embeddings_ usando la clase `AutoModel`. Todo lo que tenemos que hacer es escoger un punto de control adecuado para cargar el modelo. Afortunadamente, existe una librería llamada `sentence-transformers` que se especializa en crear _embeddings_. Como se describe en la [documentación](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search) de esta librería, nuestro caso de uso es un ejemplo de _búsqueda semántica asimétrica_ porque tenemos una pregunta corta cuya respuesta queremos encontrar en un documento más grande, como un comentario de un issue. La tabla de [resumen de modelos](https://www.sbert.net/docs/pretrained_models.html#model-overview) en la documentación nos indica que el punto de control `multi-qa-mpnet-base-dot-v1` tiene el mejor desempeño para la búsqueda semántica, así que lo usaremos para nuestra aplicación. También cargaremos el tokenizador usando el mismo punto de control:


**PyTorch:**

```py
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
```

Para acelerar el proceso de _embedding_, es útil ubicar el modelo y los inputs en un dispositivo GPU, así que hagámoslo:

```py
import torch

device = torch.device("cuda")
model.to(device)
```

**TensorFlow/Keras:**

```py
from transformers import AutoTokenizer, TFAutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
```

Ten en cuenta que hemos definido `from_pt=True` como un argumento del método `from_pretrained()`. Esto es porque el punto de control `multi-qa-mpnet-base-dot-v1` sólo tiene pesos de PyTorch, asi que usar `from_pt=True` los va a convertir automáticamente al formato TensorFlow. Como puedes ver, ¡es múy fácil cambiar entre frameworks usando 🤗 Transformers!


Como mencionamos con anterioridad, queremos representar cada entrada en el corpus de issues de GitHub como un vector individual, así que necesitamos agrupar o promediar nuestros _embeddings_ de tokes de alguna manera. Un abordaje popular es ejecutar *CLS pooling* en los outputs de nuestro modelo, donde simplemente vamos a recolectar el último estado oculto para el token especial `[CLS]`. La siguiente función nos ayudará con esto:

```py
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
```

Ahora crearemos una función que va a tokenizar una lista de documentos, ubicar los tensores en la GPU, alimentarlos al modelo y aplicar CLS pooling a los outputs:


**PyTorch:**

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

Podemos probar que la función sirve al pasarle la primera entrada de texto en el corpus e inspeccionando la forma de la salida:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
torch.Size([1, 768])
```

¡Hemos convertido la primera entrada del corpus en un vector de 768 dimensiones! Ahora podemos usar `Dataset.map()` para aplicar nuestra función `get_embeddings()` a cada fila del corpus, así que creemos una columna `embeddings` así:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```

**TensorFlow/Keras:**

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

Podemos probar que la función sirve al pasarle la primera entrada de texto en el corpus e inspeccionando la forma de la salida:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
TensorShape([1, 768])
```

¡Hemos convertido la primera entrada del corpus en un vector de 768 dimensiones! Ahora podemos usar `Dataset.map()` para aplicar nuestra función `get_embeddings()` a cada fila del corpus, así que creemos una columna `embeddings` así:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)
```


Los _embeddings_ se han convertido en arrays de NumPy, esto es porque 🤗 Datasets los necesita en este formato cuando queremos indexarlos con FAISS, que es lo que haremos a continuación.

## Usando FAISS para una búsqueda eficiente por similaridad

Ahora que tenemos un dataset de embeddings, necesitamos una manera de buscar sobre ellos. Para hacerlo, usaremos una estructura especial de datos en 🤗 Datasets llamada _índice FAISS_. [FAISS] (https://faiss.ai/) (siglas para _Facebook AI Similarity Search_) es una librería que contiene algoritmos eficientes para buscar y agrupar rápidamente vectores de _embeddings_.

La idea básica detrás de FAISS es que crea una estructura especial de datos, llamada _índice_, que te permite encontrar cuáles embeddings son parecidos a un _embedding_ de entrada. La creación de un índice FAISS en 🤗 Datasets es muy simple: usamos la función `Dataset.add_faiss_index()` y especificamos cuál columna del dataset queremos indexar:

```py
embeddings_dataset.add_faiss_index(column="embeddings")
```

Ahora podemos hacer búsquedas sobre este índice al hacer una búsqueda del vecino más cercano con la función `Dataset.get_nearest_examples()`. Probémoslo al hacer el _embedding_ de una pregunta de la siguiente manera:


**PyTorch:**

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape
```

```python out
torch.Size([1, 768])
```

**TensorFlow/Keras:**

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).numpy()
question_embedding.shape
```

```python out
(1, 768)
```


Tal como en los documentos, ahora tenemos un vector de 768 dimensiones que representa la pregunta, que podemos comparar con el corpus entero para encontrar los _embeddings_ más parecidos:

```py
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
```

La función `Dataset.get_nearest_examples()` devuelve una tupla de puntajes que calcula un ranking de la coincidencia entre la pregunta y el documento, así como un conjunto correspondiente de muestras (en este caso, los 5 mejores resultados). Recojámoslos en un `pandas.DataFrame` para ordenarlos fácilmente:

```py
import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
```

Podemos iterar sobre las primeras filas para ver qué tanto coincide la pregunta con los comentarios disponibles:

```py
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
```

```python out
"""
COMMENT: Requiring online connection is a deal breaker in some cases unfortunately so it'd be great if offline mode is added similar to how `transformers` loads models offline fine.

@mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?
SCORE: 25.505046844482422
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: The local dataset builders (csv, text , json and pandas) are now part of the `datasets` package since #1726 :)
You can now use them offline
\`\`\`python
datasets = load_dataset("text", data_files=data_files)
\`\`\`

We'll do a new release soon
SCORE: 24.555509567260742
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: I opened a PR that allows to reload modules that have already been loaded once even if there's no internet.

Let me know if you know other ways that can make the offline mode experience better. I'd be happy to add them :)

I already note the "freeze" modules option, to prevent local modules updates. It would be a cool feature.

----------

> @mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?

Indeed `load_dataset` allows to load remote dataset script (squad, glue, etc.) but also you own local ones.
For example if you have a dataset script at `./my_dataset/my_dataset.py` then you can do
\`\`\`python
load_dataset("./my_dataset")
\`\`\`
and the dataset script will generate your dataset once and for all.

----------

About I'm looking into having `csv`, `json`, `text`, `pandas` dataset builders already included in the `datasets` package, so that they are available offline by default, as opposed to the other datasets that require the script to be downloaded.
cf #1724
SCORE: 24.14896583557129
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: > here is my way to load a dataset offline, but it **requires** an online machine
>
> 1. (online machine)
>
> ```
>
> import datasets
>
> data = datasets.load_dataset(...)
>
> data.save_to_disk(/YOUR/DATASET/DIR)
>
> ```
>
> 2. copy the dir from online to the offline machine
>
> 3. (offline machine)
>
> ```
>
> import datasets
>
> data = datasets.load_from_disk(/SAVED/DATA/DIR)
>
> ```
>
>
>
> HTH.


SCORE: 22.893993377685547
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: here is my way to load a dataset offline, but it **requires** an online machine
1. (online machine)
\`\`\`
import datasets
data = datasets.load_dataset(...)
data.save_to_disk(/YOUR/DATASET/DIR)
\`\`\`
2. copy the dir from online to the offline machine
3. (offline machine)
\`\`\`
import datasets
data = datasets.load_from_disk(/SAVED/DATA/DIR)
\`\`\`

HTH.
SCORE: 22.406635284423828
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================
"""
```

¡No está mal! El segundo comentario parece responder la pregunta.

> [!TIP]
> ✏️ **¡Inténtalo!** Crea tu propia pregunta y prueba si puedes encontrar una respuesta en los documentos devueltos. Puede que tengas que incrementar el parámetro `k` en `Dataset.get_nearest_examples()` para aumentar la búsqueda.

---

# 🤗 Datasets, ¡listo!


Bueno, ese fue un gran tour de la librería 🤗 Datasets. ¡Felicitaciones por llegar hasta aquí! Con el conocimiento que adquiriste en este capítulo, deberías ser capaz de:

- Cargar datasets de cualquier parte, sea del Hub de Hugging Face, tu computador o un servidor remoto en tu compañía.
- Preparar tus datos usando una combinación de las funciones `Dataset.map()` y `Dataset.filter()`.
- Cambiar rápidamente entre formatos de datos como Pandas y NumPy usando `Dataset.set_format()`.
- Crear tu propio dataset y subirlo al Hub de Hugging Face.
- Procesar tus documentos usando un modelo de Transformer y construir un motor de búsqueda semántica usando FAISS.

En el [Capítulo 7](/course/chapter7) pondremos todo esto en práctica cuando veamos a profundidad las tareas de PLN en las que son buenos los modelos de Transformers. Antes de seguir, ¡es hora de poner a prueba tu conocimiento de 🤗 Datasets con un quiz!


---



# Quiz de final de capítulo


¡Vimos muchas cosas en este capítulo! No te preocupes si no te quedaron claros todos los detalles; los siguientes capítulos te ayudarán a entender cómo funcionan las cosas internamente.

Antes de seguir, probemos lo que aprendiste en este capítulo:

### 1. ¿Desde qué ubicaciones te permite cargar datasets la función `load_dataset()` en 🤗 Datasets?


- Localmente, e.g. en tu computador
- El Hub de Hugging Face
- Un servidor remoto


### 2. Supón que cargas una de las tareas de GLUE así:

```py
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")
```

¿Cuál de los siguientes comandos a a producir una muestra aleatoria de 50 elementos de `dataset`?


- `dataset.sample(50)`
- `dataset.shuffle().select(range(50))`
- `dataset.select(range(50)).shuffle()`


### 3. Supón que tienes un dataset sobre mascotas llamado `pets_dataset`, que tiene una columna `name` que contiene el nombre de cada mascota. ¿Cuál de los siguientes acercamientos te permitiría filtrar el dataset para todas las mascotas cuyos nombres comienzan con la letra "L"?


- <code>pets_dataset.filter(lambda x : x[
- <code>pets_dataset.filter(lambda x[
- Crear una función como <code>def filter_names(x): return x[


### 4. ¿Qué es la proyección en memoria (_memory mapping_)?


- Un mapeo entre la RAM de la CPU y la GPU
- Un mapeo entre la RAM y el sistema de almacenamiento de archivos
- Un mapeo entre dos archivos en el cache de 🤗 Datasets


### 5. ¿Cuáles son los principales beneficios de la proyección en memoria?


- Acceder a los archivos proyectados en memoria es más rápido que leerlos de o guardarlos en el disco.
- Las aplicaciones pueden acceder a segmentos de los datos en un archivo extremadamente grande sin necesidad de cargar el archivo completo en la RAM.
- Consume menos energía, así que tu batería dura más.


### 6. ¿Por qué no funciona el siguiente código?

```py
from datasets import load_dataset

dataset = load_dataset("allocine", streaming=True, split="train")
dataset[0]
```


- Intenta hacer _streaming_ de un dataset que es muy grande para caber en la RAM.
- Intenta acceder un `IterableDataset`.
- El dataset `allocine` no tiene un conjunto `train`.


### 7. ¿Cuáles son los principales beneficios de crear una tarjeta para un dataset?


- Provee información sobre el uso esperado del dataset y las tareas soportadas, para que otros en la comunidad puedan tomar una decisión informada sobre usarlo.
- Ayuda a llamar la atención a los sesgos que están presentes en un corpus.
- Aumenta las probabilidades de que otros en la comunidad usen mi dataset.


### 8. ¿Qué es la búsqueda semántica?


- Una forma de buscar coincidencias exactas entre las palabras de una pregunta y los documentos de un corpus
- Una forma de emparejar documentos entendiendo el significado contextual de una pregunta
- Una forma de mejorar la precisión de la búsqueda


### 9. Para la búsqueda semántica asimétrica, usualmente tienes:


- Una pregunta corta y un párrafo largo que responde la pregunta
- Preguntas y párrafos de una longitud similar
- Una pregunta larga y un párrafo más corto que responde la pregunta


### 10. ¿Puedo usar 🤗 Datasets para cargar datos y usarlos en otras áreas, como procesamiento de habla?


- No
- Yes



---

# 6. La librería 🤗 Tokenizers

# Introducción[[introduction]]


En el [Capítulo 3](/course/chapter3), revisamos como hacer fine-tuning a un modelo para una tarea dada. Cuando hacemos eso, usamos el mismo tokenizador con el que el modelo fue entrenado -- pero, ¿Qué hacemos cuando queremos entrenar un modelo desde cero? En estos casos, usar un tokenizador que fue entrenado en un corpus con otro dominio u otro lenguaje típicamente no es lo más óptimo. Por ejemplo un tokenizador que es entrenado en un corpus en Inglés tendrá un desempeño pobre en un corpus de textos en Japonés porque el uso de los espacios y de la puntuación es muy diferente entre los dos lenguajes.


En este capítulo, aprenderás como entrenar un tokenizador completamente nuevo en un corpus, para que luego pueda ser usado para pre-entrenar un modelo de lenguaje. Todo esto será hecho con la ayuda de la librería [🤗 Tokenizers](https://github.com/huggingface/tokenizers), la cual provee tokenizadores rápidos (_fast tokenizers_) en la librería [🤗 Transformers](https://github.com/huggingface/transformers). Miraremos de cerca todas las características que la provee la librería, y explorar cómo los tokenizadores rápidos (fast tokenizers) difieren de las versiones "lentas".

Los temas a cubrir incluyen:

* Cómo entrenar un tokenizador nuevo similar a los usados por un checkpoint dado en un nuevo corpus de texto.
* Las características especiales de los tokenizador rápidos ("fast tokenizers").
* Las diferencias entre los tres principales algoritmos de tokenización usados en PLN hoy.
* Como construir un tokenizador desde cero con la librería 🤗 Tokenizers y entrenarlo en datos. 

Las técnicas presentadas en este capítulo te prepararán para la sección en el [Capítulo 7](/course/chapter7/6) donde estudiaremos cómo crear un modelo de lenguaje para Código Fuente en Python. Comenzaremos en primer lugar revisando qué significa "entrenar" un tokenizador.

---

# Entrenar un nuevo tokenizador a partir de uno existente[[training-a-new-tokenizer-from-an-old-one]]


Si un modelo de lenguaje no está disponible en el lenguaje en el que estás interesado, o si el corpus es muy diferente del lenguaje original en el que el modelo de lenguaje fue entrenado, es muy probable que quieras reentrenar el modelo desde cero utilizando un tokenizador adaptado a tus datos. Eso requerirá entrenar un tokenizador nuevo en tu conjunto de datos. Pero, ¿Qué significa eso exactamente? Cuando revisamos los tokenizadores por primera vez en el [Capítulo 2](/course/chapter2), vimos que la mayoría de los modelos basados en Transformers usan un algoritmo de _tokenización basado en subpalabras_. Para identificar qué subpalabras son de interés y ocurren más frecuentemente en el corpus deseado, el tokenizador necesita mirar de manera profunda todo el texto en el corpus -- un proceso al que llamamos *entrenamiento*. Las reglas exactas que gobiernan este entrenamiento dependen en el tipo de tokenizador usado, y revisaremos los 3 algoritmos principales más tarde en el capítulo.


**Video:** [Ver en YouTube](https://youtu.be/DJimQynXZsQ)


> [!WARNING]
> ⚠️ ¡Entrenar un tokenizador no es lo mismo que entrenar un modelo! Entrenar un modelo utiliza `stochastic gradient descent` para minimizar la pérdida (`loss`) en cada lote (`batch`). Es un proceso aleatorio por naturaleza (lo que signifiva que hay que fijar semillas para poder obterner los mismos resultados cuando se realiza el mismo entrenamiento dos veces). Entrenar un tokenizador es un proceso estadístico que intenta identificar cuales son las mejores subpalabras para un corpus dado, y las reglas exactas para elegir estas subpalabras dependen del algoritmo de tokenización. Es un proceso deterministico, lo que significa que siempre se obtienen los mismos resultados al entrenar el mismo algoritmo en el mismo corpus.

## Ensamblando un Corpus[[assembling-a-corpus]]

Hay una API muy simple en 🤗 Transformers que se puede usar para entrenar un nuevo tokenizador con las mismas características que uno existente: `AutoTokenizer.train_new_from_iterator()`. Para verlo en acción, digamos que queremos entrenar GPT-2 desde cero, pero en lenguaje distinto al Inglés. Nuestra primera tarea será reunir muchos datos en ese lenguaje en un corpus de entrenamiento. Para proveer ejemplos que todos serán capaces de entender no usaremos un lenguaje como el Ruso o el Chino, sino uno versión del inglés más especializado: Código en Python.

La librería [🤗 Datasets](https://github.com/huggingface/datasets) nos puede ayudar a ensamblar un corpus de código fuente en Python. Usaremos la típica función `load_dataset()` para descargar y cachear el conjunto de datos [CodeSearchNet](https://huggingface.co/datasets/code_search_net). Este conjunto de datos fue creado para el [CodeSearchNet challenge](https://wandb.ai/github/CodeSearchNet/benchmark) y contiene millones de funciones de librerías open source en GitHub en varios lenguajes de programación. Aquí cargaremos la parte del conjunto de datos que está en Python:

```py
from datasets import load_dataset

# Esto puede tomar varios minutos para cargarse, así que ¡Agarra un té o un café mientras esperas!
raw_datasets = load_dataset("code_search_net", "python")
```
Podemos echar un vistazo a la porción de entrenamiento para ver a qué columnas tenemos acceso:

```py
raw_datasets["train"]
```

```python out
Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})
```

Podemos ver que el conjunto de datos separa los docstrings del código y sugiere una tokenización de ambos. Acá, sólo utilizaremos la columna `whole_func_string` para entrenar nuestro tokenizador. Podemos mirar un ejemplo de estas funciones utilizando algún índice en la porción de "train".

```py
print(raw_datasets["train"][123456]["whole_func_string"])
```
lo cual debería imprimir lo siguiente:

```out
def handle_simple_responses(
      self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
    """Accepts normal responses from the device.

    Args:
      timeout_ms: Timeout in milliseconds to wait for each response.
      info_cb: Optional callback for text sent from the bootloader.

    Returns:
      OKAY packet's message.
    """
    return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)
```

Lo primero que necesitamos hacer es transformar el dataset en un _iterador_ de listas de textos -- por ejemplo, una lista de listas de textos. utilizar listas de textos permitirá que nuestro tokenizador vaya más rápido (entrenar en batches de textos en vez de procesar textos de manera individual uno por uno), y debería ser un iterador si queremos evitar tener cargar todo en memoria de una sola vez. Si tu corpus es gigante, querrás tomar ventaja del hecho que 🤗 Datasets no carga todo en RAM sino que almacena los elementos del conjunto de datos en disco.
Hacer lo siguiente debería crear una lista de listas de 1000 textos cada una, pero cargando todo en memoria:

```py
# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]
```

Al usar un generador de Python, podemos evitar que Python cargue todo en memoria hasta que sea realmente necesario. Para crear dicho generador, solo necesitas reemplazar los corchetes con paréntesis:

```py
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```
Esta línea de código no trae ningún elemento del conjunto de datos; sólo crea un objeto que se puede usar en Python con un ciclo `for`. Los textos sólo serán cargados cuando los necesites (es decir, cuando estás un paso del ciclo `for` que los requiera), y sólo 1000 textos a la vez serán cargados. De eso forma no agotarás toda tu memoria incluso si procesas un conjunto de datos gigante. 

El problema con un objeto generador es que sólo se puede usar una vez. Entonces en vea que el siguiente código nos entregue una lista de los primeros 10 dígitos dos veces:

```py
gen = (i for i in range(10))
print(list(gen))
print(list(gen))
```
Nos lo entrega una vez, y luego una lista vacía:

```python out
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[]
```

Es por eso que definimos una función que retorne un generador:

```py
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()
```
También puedes definir un generador dentro de un ciclo `for`utilizando el comando `yield`:

```py
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
```

el cual producirá el mismo generador anterior, pero también permitiendo usar lógicas más complejas de las que se puede hacer en un `list comprehension`.

## Entrenar un nuevo Tokenizador[[training-a-new-tokenizer]]

Ahora que tenemos nuestro corpus en la forma de un iterador de lotes de textos, estamos listos para entrenar un nuevo tokenizador. Para hacer esto, primero tenemos que cargar el tokenizador que queremos utilizar con nuestro modelo (en este caso, GPT-2):

```py
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
Aunque vamos a entrenar un nuevo tokenizador, es una buena idea hacer esto para evitar comenzar de cero completamente. De esta manera, no tendremos que especificar nada acerca del algoritmo de tokenización o de los tokens especiales que queremos usar; nuestro tokenizador será exactamente el mismo que GPT-2, y lo único que cambiará será el vocabulario, el cuál será determinado por el entrenamiento en nuestro corpus. 

Primero, echemos un vistazo a cómo este tokenizador tratará una función de ejemplo:

```py
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokens
```

```python out
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo',
 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

Este tokenizador tiene algunos símbolos especiales como `Ġ` y `Ċ`, lo cual denota espacios y nuevas líneas (saltos de líneas) respectivamente. Como podemos ver, esto no es muy eficiente: el tokenizador retorna tokens individuales para cada espacio, cuando debería agrupar los niveles de indentación (dado que tener grupos de cuatro u ocho espacios va a ser muy común en el uso de código). Además separa el nombre de la función de manera un poco extraña al no estar acostumbrado a ver palabras separadas con el caracter `_`.

Entrenemos nuestro nuevo tokenizador y veamos si resuelve nuestros problemas. Para esto usaremos el método `train_new_from_iterator()`:

```py
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```
Este comando puede tomar tiempo si tu corpus es muy largo, pero para este conjunto de datos de 1.6 GB de textos es muy rápido (1 minuto 16 segundos en un AMD Ryzen 9 3900X CPU con 12 núcleos).

Nota que `AutoTokenizer.train_new_from_iterator()` sólo funciona si el tokenizador que estás usando es un tokenizador rápido (_fast tokenizer_). Cómo verás en la siguiente sección, la librería 🤗 Transformers contiene 2 tipos de tokenizadores: algunos están escritos puramente en Python y otros (los rápidos) están respaldados por la librería 🤗 Tokenizers, los cuales están escritos en lenguaje de programación [Rust](https://www.rust-lang.org). Python es el lenguaje mayormente usado en ciencia de datos y aplicaciones de deep learning, pero cuando algo necesita ser paralelizado para ser rápido, tiene que ser escrito en otro lenguaje. Por ejemplo, las multiplicaciones matriciales que están en el corazón de los cómputos de un modelo están escritos en CUDA, una librería optimizada en C para GPUs. del computation are written in CUDA, an optimized C library for GPUs.

Entrenar un nuevo tokenizador en Python puro sería insoportablemente lento, razón pr la cual desarrollamos la librería 🤗 Tokenizers. Notar que de la misma manera que no tuviste que aprender el lenguaje CUDA para ser capaz de ejecutar tu modelo en un barch de inputs en una GPU, no necesitarás aprender Rust para usar los tokenizadores rápidos (_fast tokenizers_). La librería 🤗 Tokenizers provee bindings en Python para muchos métodos que internamente llaman trozos de código en Rust; por ejemplo, para paralelizar el entrenamiento de un nuevo tokenizador o, como vimos en el [Capítulo 3](/course/chapter3), la tokenización de un batch de inputs. 

La mayoría de los modelos Transformers tienen un tokenizador rápido (_Fast Tokenizer_) disponible (hay algunas excepciones que se pueden revisar [acá](https://huggingface.co/transformers/#supported-frameworks)), y la API `AutoTokenizer` siempre seleccionar un tokenizador rápido para ti en caso de estar disponible. En la siguiente sección echaremos un vistazo a algunas de las características especiales que tienen los tokenizadores rápidos, los cuales serán realmente útiles para tareas como clasificación de tokens y question answering. Antes de sumergirnos en eso, probemos nuestro tokenizador recién entrenado en nuestro ejemplo previo:

```py
tokens = tokenizer.tokenize(example)
tokens
```

```python out
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

Acá nuevamente vemos los símbolos especiales `Ġ` y `Ċ` que denotan espacios y nuevas líneas (saltos de líneas), pero también podemos ver que nuestro tokenizador aprendió algunos tokens que son altamente específicos para el corpus de funciones en Python: por ejemplo, está el token `ĊĠĠĠ` que representa una indentación y un token `Ġ"""` que representan la triple comilla para comenzar un docstring. El tokenizador también divide correctamente los nombres de funciones usando `_`. Esta es una representación más compacta ya que utilizar un tokenizador común y corriente en inglés en el mismo ejemplo nos dara una oración más larga:


```py
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))
```

```python out
27
36
```
Echemos un vistazo al siguiente ejemplo:

```python
example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokenizer.tokenize(example)
```

```python out
['class', 'ĠLinear', 'Layer', '():', 'ĊĠĠĠ', 'Ġdef', 'Ġ__', 'init', '__(', 'self', ',', 'Ġinput', '_', 'size', ',',
 'Ġoutput', '_', 'size', '):', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'weight', 'Ġ=', 'Ġtorch', '.', 'randn', '(', 'input', '_',
 'size', ',', 'Ġoutput', '_', 'size', ')', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'bias', 'Ġ=', 'Ġtorch', '.', 'zeros', '(',
 'output', '_', 'size', ')', 'ĊĊĠĠĠ', 'Ġdef', 'Ġ__', 'call', '__(', 'self', ',', 'Ġx', '):', 'ĊĠĠĠĠĠĠĠ',
 'Ġreturn', 'Ġx', 'Ġ@', 'Ġself', '.', 'weights', 'Ġ+', 'Ġself', '.', 'bias', 'ĊĠĠĠĠ']
```

En adición al token correspondiente a la indentación, también podemos ver un token para la doble indentación: `ĊĠĠĠĠĠĠĠ`. Palabras espaciales del lenguaje Python como `class`, `init`, `call`, `self`, and `return` son tokenizadas como un sólo token y podemos ver que además de dividir en `_` y `.`, el tokenizador correctamente divide incluso en nombres que usan camel-case: `LinearLayer` es tokenizado como `["ĠLinear", "Layer"]`.

## Guardar el Tokenizador[[saving-the-tokenizer]]


Para asegurarnos que podemos usar el tokenizador más tarde, necesitamos guardar nuestro nuevo tokenizador. Al igual que los modelos, esto se hace con el método `save_pretrained()`. 

```py
tokenizer.save_pretrained("code-search-net-tokenizer")
```

Esto creará una nueva carpeta llamada *code-search-net-tokenizer*, la cual contendrá todos los archivos que el tokenizador necesita para ser cargado. Si quieres compartir el tokenizador con tus colegas y amigos, puedes subirlo al Hub logeando en tu cuenta. Si estás trabajando en notebooks, hay una función conveniente para ayudarte a hacer esto:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Esto mostrará un widget donde puedes ingresar tus credenciales de Hugging Face. En caso de no estar usando un notebook, puedes escribir la siguiente línea en tu terminal:

```bash
huggingface-cli login
```

Una vez logueado puedes enviar tu tokenizador al Hub ejecutando el siguiente comando::

```py
tokenizer.push_to_hub("code-search-net-tokenizer")
```
Esto creará un nuevo repositorio en tu namespace con el nombre `code-search-net-tokenizer`, conteniendo el archivo del tokenizador. Luego puedes cargar tu tokenizador desde donde quieras utilizando método `from_pretrained()`.

```py
# Replace "huggingface-course" below with your actual namespace to use your own tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```
Ya estás listo para entrenar un modelo de lenguaje desde cero y hacer fine-tuning en la tarea que desees. Llegaremos a eso en el [Capítulo 7](/course/chapter7), pero primero en el resto del capítulo miraremos más de cerca los tokenizadores rápidos (_Fast Tokenizers_) y explorar en detalle lo que pasa en realidad pasa cuando llamamos al método `train_new_from_iterator()`.


---



# Los poderes especiales de los Tokenizadores Rápidos (Fast tokenizers)[[fast-tokenizers-special-powers]]


En esta sección miraremos más de cerca las capacidades de los tokenizadores en 🤗 Transformers. Hasta ahora sólo los hemos utilizado para tokenizar las entradas o decodificar los IDs en texto, pero los tokenizadores -- especialmente los que están respaldados en la librería 🤗 Tokenizers -- pueden hacer mucho más. Para ilustrar estas características adicionales, exploraremos cómo reproducir los resultados de los pipelines de `clasificación de tokens` (al que llamamos ner) y `question-answering` que nos encontramos en el [Capítulo 1](/course/chapter1).


**Video:** [Ver en YouTube](https://youtu.be/g8quOxoqhHQ)


En la siguiente discusión, a menudo haremos la diferencia entre un tokenizador "lento" y uno "rápido". Los tokenizadores lentos son aquellos escritos en Python dentro de la librería Transformers, mientras que las versiones provistas por la librería 🤗 Tokenizers, son los que están escritos en Rust. Si recuerdas la tabla del [Capítulo 5](/course/chapter5/3) en la que se reportaron cuanto tomó a un tokenizador rápido y uno lento  tokenizar el Drug Review Dataset, ya deberías tener una idea de por qué los llamamos rápidos y lentos:


|               | Tokenizador Rápido | Tokenizador Lento
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s

> [!WARNING]
> ⚠️ Al tokenizar una sóla oración, no siempre verás una diferencia de velocidad entre la versión lenta y la rápida del mismo tokenizador. De hecho, las versión rápida podría incluso ser más lenta! Es sólo cuando se tokenizan montones de textos en paralelos al mismo tiempo que serás capaz de ver claramente la diferencia.

## Codificación en Lotes (Batch Encoding)[[batch-encoding]]


**Video:** [Ver en YouTube](https://youtu.be/3umI3tm27Vw)


La salida de un tokenizador no siempre un simple diccionario; lo que se obtiene en realidad es un objeto especial `BatchEncoding`. Es una subclase de un diccionario (razón por la cual pudimos indexar el resultado sin ningún problema anteriormente), pero con métodos adicionales que son mayormente usados por los tokenizadores rápidos (_Fast Tokenizers_).

Además de sus capacidad en paralelización, la funcionalidad clave de un tokenizador rápido es que siempre llevan registro de la porción de texto de la cual los tokens finales provienen -- una característica llamada *offset mapping*. Esto permite la capacidad de mapear cada palabra con el token generado o mapear cada caracter del texto original con el token respectivo y viceversa. 

Echemos un vistazo a un ejemplo:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
```
Como se mencionó previamente, obtenemos un objeto de tipo `BatchEncoding` como salida del tokenizador:

```python out
<class 'transformers.tokenization_utils_base.BatchEncoding'>
```

Dado que la clase  `AutoTokenizer` escoge un tokenizador rápido por defecto, podemos usar los métodos adicionales que este objeto `BatchEncoding` provee. Tenemos dos manera de chequear si el tokenizador es rápido o lento. Podemos chequear el atributo `is_fast` del tokenizador:

```python
tokenizer.is_fast
```

```python out
True
```
o chequear el mismo atributo de nuestro `encoding`:

```python
encoding.is_fast
```

```python out
True
```

Veamos lo que un tokenizador rápido nos permite hacer. Primero podemos acceder a los tokens sin tener que convertir los IDs a tokens:

```py
encoding.tokens()
```

```python out
['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in',
 'Brooklyn', '.', '[SEP]']
```

En este caso el token con índice 5 is `##yl`, el cual es parte de la palabra "Sylvain" en la oración original. Podemos también utilizar el método `word_ids()` para obtener el índice de la palabra de la que cada token proviene:

```py
encoding.word_ids()
```

```python out
[None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]
```

Podemos ver que los tokens especiales del tokenizador `[CLS]` y `[SEP]` están mapeados a `None`, y que cada token está mapeado a la palabra de la cual se origina. Esto es especialmente útil para determinar si el token está al inicio de la palabra o si dos tokens están en la misma palabra. POdríamos confiar en el prefijo `[CLS]` and `[SEP]` para eso, pero eso sólo funciona para tokenizadores tipo BERT; este método funciona para cualquier tipo de tokenizador mientras sea de tipo rápido. En el próximo capítulo, veremos como podemos usar esta capacidad para aplicar etiquetas para cada palabra de manera apropiada en tareas como Reconocimiento de Entidades (Named Entity Recognition NER), y etiquetado de partes de discurso (part-of-speech POS tagging). También podemos usarlo para enmascarar todos los tokens que provienen de la misma palabra en masked language modeling (una técnica llamada _whole word masking_).

> [!TIP]
> La noción de qué es una palabra es complicada. Por ejemplo "I'll" (la contracción de "I will" en inglés) ¿cuenta como una o dos palabras? De hecho depende del tokenizador y la operación de pretokenización que aplica. Algunos tokenizadores sólo separan en espacios, por lo que considerarán esto como una sóla palabra. Otros utilizan puntuación por sobre los espacios, por lo que lo considerarán como dos palabras. 
>
> ✏️ **Inténtalo!** Crea un tokenizador a partir de los checkpoints `bert-base-cased` y `roberta-base` y tokeniza con ellos "81s". ¿Qué observas? Cuál son los IDs de la palabra?

De manera similar está el método `sentence_ids()` que podemos utilizar para mapear un token a la oración de la cuál proviene (aunque en este caso el `token_type_ids` retornado por el tokenizador puede darnos la misma información).

Finalmente, podemos mapear cualquier palabra o token a los caracteres originales del texto, y viceversa, utilizando los métodos `word_to_chars()` o `token_to_chars()` y los métodos `char_to_word()` o `char_to_token()`. Por ejemplo el método `word_ids()` nos dijo que `##yl` es parte de la palabra con índice 3, pero qué palabra es en la oración? Podemos averiguarlo así:

```py
start, end = encoding.word_to_chars(3)
example[start:end]
```

```python out
Sylvain
```

Como mencionamos previamente, todo esto funciona gracias al hecho de que los tokenizadores rápidos llevan registro de la porción de texto del que cada token proviene en una lista de *offsets*. Para ilustrar sus usos, a continuación mostraremos como replicar los resultados del pipeline de `clasificación de tokens` de manera manual.

> [!TIP]
> ✏️ **Inténtalo!** Crea tu propio texto de ejemplo y ve si puedes entender qué tokens están asociados con el ID de palabra, y también cómo extraer los caracteres para una palabra. Como bonus, intenta usar dos oraciones como entrada/input y ve si los IDs de oraciones te hacen sentido.

## Dentro del Pipeline de `clasificación de tokens`[[inside-the-token-classification-pipeline]]

En el [Capítulo 1](/course/chapter1) tuvimos nuestra primera probada aplicando NER -- donde la tarea es identificar qué partes del texto corresponden a entidades como personas, locaciones, u organizaciones -- con la función `pipeline()` de la librería 🤗 Transformers. Luego en el [Capítulo 2](/course/chapter2), vimos como un pipeline agrupa las tres etapas necesarias para obtener predicciones desde un texto crudo: tokenización, pasar los inputs a través del modelo, y post-procesamiento. Las primeras dos etapas en el pipeline de `clasificación de tokens` son las mismas que en otros pipelines, pero el post-procesamiento es un poco más complejo -- ×+/¡veamos cómo!


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/0E7ltQB7fM8)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/PrX4CjrVnNc)


### Obteniendo los resultados base con el pipeline[[getting-the-base-results-with-the-pipeline]]

Primero, agarremos un pipeline de clasificación de tokens para poder tener resultados que podemos comparar manualmente. El usado por defecto es [`dbmdz/bert-large-cased-finetuned-conll03-english`](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english); el que realiza NER en oraciones:

```py
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

El modelo indentificó apropiadamente cada token generado por "Sylvain" como una persona, cada token generado por "Hugging Face" como una organización y el token "Brooklyn" como una locación. Podemos pedirle también al pipeline que agrupe los tokens que corresponden a la misma identidad:

```py
from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

La estrategia de agregación (`aggregation_strategy`) elegida cambiará los puntajes calculados para cada entidad agrupada. Con `"simple"` el puntaje es la media los puntajes de cada token en la entidad dada: por ejemplo, el puntaje de "Sylvain" es la media de los puntajes que vimos en el ejemplo previo para los tokens `S`, `##yl`, `##va`, y `##in`. Otras estrategias disponibles son:

- `"first"`, donde el puntaje de cada entidad es el puntaje del primer token de la entidad (para el caso de "Sylvain" sería 0.9923828, el puntaje del token `S`)
- `"max"`, donde el puntaje de cada entidad es el puntaje máximo de los tokens en esa entidad (para el caso de "Hugging Face" sería 0.98879766, el puntaje de "Face")
- `"average"`, donde el puntaje de cada entidad es el promedio de los puntajes de las palabras que componen la entidad (para el caso de "Sylvain" no habría diferencia con la estrategia "simple", pero "Hugging Face" tendría un puntaje de 0.9819, el promedio de los puntajes para "Hugging", 0.975, y "Face", 0.98879)

Ahora veamos como obtener estos resultados sin utilizar la función `pipeline()`!

### De los inputs a las predicciones[[from-inputs-to-predictions]]


**PyTorch:**

Primero necesitamos tokenizar nuestro input y pasarlo a través del modelo. Esto es exactamente lo que se hace en el [Capítulo 2](/course/chapter2); instanciamos el tokenizador y el modelo usando las clases `AutoXxx` y luego los usamos en nuestro ejemplo:


```py
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
```

Dado que estamos usando acá `AutoModelForTokenClassification`, obtenemos un conjunto de logits para cada token en la secuencia de entrada:

```py
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
```

```python out
torch.Size([1, 19])
torch.Size([1, 19, 9])
```

**TensorFlow/Keras:**

Primero necesitamos tokenizar nuestro input y pasarlo por nuestro modelo. Esto es exactamente lo que se hace en el [Capítulo 2](/course/chapter2); instanciamos el tokenizador y el modelo usando las clases `TFAutoXxx` y luego los usamos en nuestro ejemplo:

```py
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="tf")
outputs = model(**inputs)
```
Dado que estamos usando acá `TFAutoModelForTokenClassification`, obtenemos un conjunto de logits para cada token en la secuencia de entrada:

```py
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
```

```python out
(1, 19)
(1, 19, 9)
```


Tenemos un lote de 1 secuencia con 19 tokens y el modelo tiene 9 etiquetas diferentes, por lo que la salida del modelo tiene dimensiones 1 x 19 x 9. Al igual que el pipeline de clasificación de texto, usamos la función softmax para convertir esos logits en probabilidades, y tomamos el argmax para obtener las predicciones (notar que podemos tomar el argmax de los logits directamente porque el softmax no cambia el orden):


**PyTorch:**

```py
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
```

**TensorFlow/Keras:**

```py
import tensorflow as tf

probabilities = tf.math.softmax(outputs.logits, axis=-1)[0]
probabilities = probabilities.numpy().tolist()
predictions = tf.math.argmax(outputs.logits, axis=-1)[0]
predictions = predictions.numpy().tolist()
print(predictions)
```


```python out
[0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
```

El atributo `model.config.id2label` contiene el mapeo de los índices con las etiquetas para que podemos hacer sentido de las predicciones:

```py
model.config.id2label
```

```python out
{0: 'O',
 1: 'B-MISC',
 2: 'I-MISC',
 3: 'B-PER',
 4: 'I-PER',
 5: 'B-ORG',
 6: 'I-ORG',
 7: 'B-LOC',
 8: 'I-LOC'}
```

Como vimos antes, hay 9 etiquetas: `0` es la etiqueta para los tokens que no tienen ningúna entidad (proviene del inglés "outside"), y luego tenemos dos etiquetas para cada tipo de entidad (misceláneo, persona, organización, y locación). La etiqueta `B-XXX` indica que el token is el inicio de la entidad `XXX` y la etiqueta `I-XXX` indica que el token está dentro de la entidad `XXX`. For ejemplo, en el ejemplo actual esperaríamos que nuestro modelo clasificará el token `S` como `B-PER` (inicio de la entidad persona), y los tokens `##yl`, `##va` y `##in` como `I-PER` (dentro de la entidad persona).

Podrías pensar que el modelo está equivocado en este caso ya que entregó la etiqueta `I-PER` a los 4 tokens, pero eso no es completamente cierto. En realidad hay 4 formatos par esas etiquetas `B-` y `I-`: *I0B1* y *I0B2*. El formato I0B2 (abajo en rosado), es el que presentamos, mientras que en el formato I0B1 (en azul), las etiquetas de comenzando con `B-` son sólo utilizadas para separar dos entidades adyacentes del mismo tipo. Al modelo que estamos usando se le hizo fine-tune en un conjunto de datos utilizando ese formato, lo cual explica por qué asigna la etiqueta `I-PER` al token `S`.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/IOB_versions.svg" alt="IOB1 vs IOB2 format"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/IOB_versions-dark.svg" alt="IOB1 vs IOB2 format"/>
</div>

Con este mapa, estamos listos para reproducir (de manera casi completa) los resultados del primer pipeline -- basta con tomar los puntajes y etiquetas de cada token que no fue clasificado como `0`:

```py
results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

print(results)
```

```python out
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S'},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl'},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va'},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in'},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu'},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging'},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face'},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn'}]
```

Esto es muy similar a lo que teníamos antes, con una excepción: el pipeline también nos dió información acerca del `inicio` y el `final` de cada entidad en la oración original. Aquí es donde nuestro mapeo de offsets entrarán en juego. Para obtener los offsets, sólo tenemos que fijar la opción `return_offsets_mapping=True` cuando apliquemos el tokenizador a nuestros inputs:

```py
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
```

```python out
[(0, 0), (0, 2), (3, 7), (8, 10), (11, 12), (12, 14), (14, 16), (16, 18), (19, 22), (23, 24), (25, 29), (30, 32),
 (33, 35), (35, 40), (41, 45), (46, 48), (49, 57), (57, 58), (0, 0)]
```

Cada tupla es la porción de texto correspondiente a cada token, donde `(0, 0)` está reservado para los tokens especiales. Vimos antes que el token con índice 5 is `##yl`, el cual tiene como offsets `(12, 14)`, Si tomamos los trozos correspondientes en nuestro ejemplo:


```py
example[12:14]
```

obtenemos la porción apropiada sin los `##`:


```python out
yl
```

Usando esto, ahora podemos completar los resultados previos:

```py
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)
```

```python out
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

Esto es lo mismo que obtuvimos en el primer pipeline!

### Agrupando Entidades[[grouping-entities]]

Usar los offsets para determinar las llaves de inicio y fin para cada entidad is útil, pero esa información no es estrictamente necesaria. Cuando queremos agrupar las entidades, sin embargo, los offsets nos ahorarán un montón de código engorroso. Por ejemplo, si queremos agrupar los tokens `Hu`, `##gging`, y `Face`, podemos hacer reglas especiales que digan que los dos primeros se tienen que unir eliminando los `##`, y `Face` debería añadirse con el espacio ya que no comienza con `##` -- pero eso sólo funcionaría para este tipo particular de tokenizador. Tendríamos que escribir otro grupo de reglas para un tokenizador tipo SentencePiece (trozo de oración) tipo Byte-Pair-Encoding (codificación por par de bytes) (los que se discutirán más adelante en este capítulo).

Con estos offsets, todo ese código hecho a medida no se necesita: basta tomar la porción del texto original que comienza con el primer token y termina con el último token. En el caso de los tokens `Hu`, `##gging`, and `Face`, deberíamos empezar en el character 33 (el inicio de `Hu`) y termianr antes del caracter 45 (al final de `Face`):

```py
example[33:45]
```

```python out
Hugging Face
```

Para escribir el código encargado del post-procesamiento de las prediciones que agrupan entidades, agruparemos la entidades que son consecutivas y etiquetadas con `I-XXX`, excepto la primera, la cual puedes estar etiquetada como `B-XXX` o `I-XXX` (por lo que, dejamos de agrupar una entidad cuando nos encontramos un `0`, un nuevo tipo de entidad, o un `B-XXX` que nos dice que una entidad del mismo tipo está empezando):

```py
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Toma todos los tokens etiquetados con la etiqueta I
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # El puntaje es la media de todos los puntajes de los tokens en la entidad agrupada
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
```

Y obtenemos los mismos resultados de nuestro segundo pipeline!

```python out
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

Otro ejemplo de una tarea donde estos offsets son extremadamente útiles es question answering. Sumergirnos en ese pipeline, lo cual haremos en la siguiente sección, también nos permitirá echar un vistazo a una última característica de los tokenizadores en la librería 🤗 Transformers: lidiar con tokens desbordados (overflowing tokens) cuando truncamos una entrada/input a un largo dado. 


---



# Tokenizadores Rápidos en un Pipeline de Question-Answering[[fast-tokenizers-in-the-qa-pipeline]]


Ahora nos sumergiremos en el pipeline de `question-answering` (preguntas y respuestas) y veremos como hacer uso de los offsets para tomar la respuesta de la pregunta desde el contexto, un poco como lo que hicimos para las entidades agrupadas en la sección previa. Luego veremos como lidiar con contextos muy largos que terminan siendo truncados. Puedes saltar esta sección si no estás interesado en la tarea de pregunta y respuesta (_question answering_).


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/_wxyB3j3mk4)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/b3u8RzBCX9Y)


## Usando el pipeline de `question-answering`[[using-the-question-answering-pipeline]]

Como vimos en el [Capítulo 1](/course/chapter1), podemos usar el pipeline de `question-answering` para obtener la respuesta a una pregunta de la siguiente manera:

```py
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)
```

```python out
{'score': 0.97773,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}
```

A diferencia de otros pipelines, los cuales no pueden truncar y dividir textos que son más largos que el largo máximo aceptado por el modelo (y por lo tanto perder información al final de un documento), este pipeline puede lidiar con contextos muy largos y retornará una respuesta a la pregunta incluso si está al final.

```py
long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question_answerer(question=question, context=long_context)
```

```python out
{'score': 0.97149,
 'start': 1892,
 'end': 1919,
 'answer': 'Jax, PyTorch and TensorFlow'}
```

¡Veamos cómo hace todo esto!

## Usando un modelo para question answering[[using-a-model-for-question-answering]]

Como para cualquier otro pipeline, empezamos tokenizando nuestro input y lo envíamos a través del modelo. El punto de control (`checkpoint`) usado por defecto para el pipeline de `question-answering` es [`distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert-base-cased-distilled-squad) (el "squad" en el nombre viene del conjunto de datos en el cual se le hizo fine-tune; hablaremos más acerca del conjunto de datos SQuAD en el [Capítulo 7](/course/chapter7/7))


**PyTorch:**

```py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```

**TensorFlow/Keras:**

```py
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="tf")
outputs = model(**inputs)
```


Notar que tokenizamos nuestra y el contexto como un par, con la pregunta primero. 

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/question_tokens.svg" alt="An example of tokenization of question and context"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/question_tokens-dark.svg" alt="An example of tokenization of question and context"/>
</div>

Los modelos para question answering funcionan de manera un poco distinta de los modelos que hemos visto hasta ahora. Usando la imagen de arriba como ejemplo, el modelo ha sido entrenado para predecir el índice de los tokens al inicio de la respuesta (en este caso el 21) y el índice del token donde la respuesta termina (en este caso el 24). Esto porque estos modelos no retornar un tensor de logits sino dos: uno para los logits correspondientes al token de inicio de la respuesta, y uno para los logits correspondientes al token de término de la respuesta. Dado que en este caso tenemos un input conteniendo 66 tokens, obtenemos:

```py
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
```


**PyTorch:**

```python out
torch.Size([1, 66]) torch.Size([1, 66])
```

**TensorFlow/Keras:**

```python out
(1, 66) (1, 66)
```


Para convertir estos logits en probabilidades, aplicaremos la función softmax -- pero antes de eso, necesitamos asegurarnos que enmascaramos los índices que no son parte del contexto. Nuestro input es `[CLS] pregunta [SEP] contexto [SEP]`, por lo que necesitamos enmascarar los tokens de la pregunta como también el token `[SEP]`. Mantredemos el token `[CLS]`, ya que algunos modelos lo usan para indicar que la respuesta no está en el contexto. 

Dado que aplicaremos una softmax después, sólo necesitamos reemplazar los logits que queremos enmascarar con un número negativo muy grande. En este caso, usamos el `-10000`:


**PyTorch:**

```py
import torch

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000
```

**TensorFlow/Keras:**

```py
import tensorflow as tf

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = tf.constant(mask)[None]

start_logits = tf.where(mask, -10000, start_logits)
end_logits = tf.where(mask, -10000, end_logits)
```


Ahora que tenemos enmascarados los logits de manera apropiada correspondientes a los tokens que no queremos predecir. Podemos aplicar la softmax:


**PyTorch:**

```py
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
```

**TensorFlow/Keras:**

```py
start_probabilities = tf.math.softmax(start_logits, axis=-1)[0].numpy()
end_probabilities = tf.math.softmax(end_logits, axis=-1)[0].numpy()
```


En esta punto, podemos tomar el argmax de las probabilidades de inicio y fin -- pero podríamos terminar con un índice de inicio que es mayot que índice de término, por lo que necesitamos tomar unas pocas precauciones más. Calcularemos la probabilidad de cada posible `start_index` and `end_index` (índice de inicio y final respectivamente) donde `start_index <= end_index`, luego tomamos la tupla `(start_index, end_index)` con la probabilidad más alta. 

Asumiendo que los eventos "La respuesta comienzda en `start_index`" y "La respuesta termina en `end_index`" son independientes, la probabilidad de que la respuesta inicie en `start_index` y termine en `end_index` es:

$$\mathrm{start\_probabilities}[\mathrm{start\_index}] \times \mathrm{end\_probabilities}[\mathrm{end\_index}]$$ 

Así que para calcular todos los puntajes, necesitamos calcular todos los productos \\(\mathrm{start\_probabilities}[\mathrm{start\_index}] \times \mathrm{end\_probabilities}[\mathrm{end\_index}]\\) donde `start_index <= end_index`.

Primero calculemos todos los posibles productos:

```py
scores = start_probabilities[:, None] * end_probabilities[None, :]
```


**PyTorch:**

Luego enmascararemos los valores donde `start_index > end_index` reemplazándolos como 0 (las otras probabilidades son todos números positivos). La función `torch.triu()` retorna la parte triangular superior de el tensor 2D pasado como argumento, por lo que hará el enmascaramiento por nosotros. 

```py
scores = torch.triu(scores)
```

**TensorFlow/Keras:**

Luego enmascararemos los valores donde `start_index > end_index` reemplazándolos como 0 (las otras probabilidades son todos números positivos). La función `np.triu()` retorna la parte triangular superior de el tensor 2D pasado como argumento, por lo que hará el enmascaramiento por nosotros. 

```py
import numpy as np

scores = np.triu(scores)
```


Ahora basta con obtener el índice el máximo. Dado que Pytorch retornará el índice en el tensor aplanado, necesitamos usar las operaciones división entera `//` y módulo `%` para obtener el `start_index` y el `end_index`: 

```py
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
```

No estamos listos aún, pero al menos ya tenemos el puntaje correcto para la respuesta (puedes chequear esto comparándolo con el primer resultado en la sección previa):

```python out
0.97773
```

> [!TIP]
> ✏️ **Inténtalo!** Calcula los índices de inicio y término para las cinco respuestas más probables.

Tenemos el `start_index` y el `end_index` de la respuesta en términos de tokens, así que ahora sólo necesitamos convertirlos en los índices de caracteres en el contexto. Aquí es donde los offsets serán sumamente útiles. Podemos tomarlos y usarlos como lo hicimos en la tarea de clasificación de tokens:

```py
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]
```

Ahora sólo tenemos que dar formato a todo para tener nuestros resultados:

```py
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)
```

```python out
{'answer': 'Jax, PyTorch and TensorFlow',
 'start': 78,
 'end': 105,
 'score': 0.97773}
```

Genial! Obtuvimos lo mismo que en nuestro primer ejemplo!

> [!TIP]
> ✏️ **Inténtalo!** Usaremos los mejores puntajes calculados anteriormente para mostrar las cinco respuestas más probables. Para revisar nuestros resultados regresa al primer pipeline y agrega `top_k=5` al llamarlo.

## Manejando contextos largos[[handling-long-contexts]]

Si tratamos de tokenizar la pregunta en un contexto largo que usamos en el ejemplo previamente, tendremos un número de tokens que es más alto que el largo máximo usado en el pipeline de `question-answering` (que es 384):

```py
inputs = tokenizer(question, long_context)
print(len(inputs["input_ids"]))
```

```python out
461
```

Entonces, necesitaremos truncar nuestras entradas/inputs al largo máximo. Hay varias maneras de hacer esto, pero no queremos truncar la pregunta, sólo el contexto. Dado que el contexto es la segunda oración, usaremos la estrategia de truncamiento `"only_second"`. El problema que aparece es que la respuesta a la pregunta podría no estar en el contexto truncado. En este caso, por ejemplo, elegimos una pregunta donde la respuesta está hacia el final del contexto, y cuando truncamos la respuesta no está presente:

```py
inputs = tokenizer(question, long_context, max_length=384, truncation="only_second")
print(tokenizer.decode(inputs["input_ids"]))
```

```python out
"""
[CLS] Which deep learning libraries back [UNK] Transformers? [SEP] [UNK] Transformers : State of the Art NLP

[UNK] Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

[UNK] Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internal [SEP]
"""
```

Esto significa que el modelo le costará bastante elegir la respuesta correcta. Para corregir eso, el pipeline de `question-answering` permite separar el contexto en trozos pequeños, especificando el largo máximo. Para asegurarnos que no separemos el contexto exactamente en un lugar incorrecto donde podríamos encontrar la respuesta, también incluye algunos traslapes (overlaps) entre los trozos. 

Podemos hacer que el tokenizador (rápido o lento) haga esto por nosotros agregando `return_overflowing_tokens=True`, y podemos especificar el traslape (overlap) que queremos con el argumento `stride`. Acá un ejemplo, usando una oración corta:

```py
sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

```python out
'[CLS] This sentence is not [SEP]'
'[CLS] is not too long [SEP]'
'[CLS] too long but we [SEP]'
'[CLS] but we are going [SEP]'
'[CLS] are going to split [SEP]'
'[CLS] to split it anyway [SEP]'
'[CLS] it anyway. [SEP]'
```

Como podemos ver, la oración ha sido dividida en trozos de tal manera que cada entrada en `inputs["input_ids"] tiene a lo más 6 tokens (tendríamos que agregar relleno (`padding`) en el último trozo para tener el mismo largo que los otros) y hay traslape (overlap) de 2 tokens entre cada uno de los trozos. 

Miremos de cerca el resultado de la tokenización:

```py
print(inputs.keys())
```

```python out
dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
```

Como se esperaba, obtenemos los IDs de entrada y una máscara de atención (attention mask). La última clave, `overflow_to_sample_mapping`, es un mapa que nos dice a qué oraciones corresponde cada resultado -- en este caso tenemos 7 resultados, todos provenientes de la (única) oración que le pasamos al tokenizador:

```py
print(inputs["overflow_to_sample_mapping"])
```

```python out
[0, 0, 0, 0, 0, 0, 0]
```

Esto es más útil cuando tokenizamos varias oraciones juntas. Por ejemplo así:

```py
sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

print(inputs["overflow_to_sample_mapping"])
```

obtenemos:

```python out
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
```

lo que significa que la primera oración está dividida en 7 trozos igual que antes, y los siguientes 4 trozos vienen de la segunda oración.

Ahora volvamos a nuestro contexto largo. Por defecto el pipeline de `question-answering` usa un largo máximo de 384, como mencionamos antes, y un stride de 128, lo que corresponde a la manera en la que al modelo se le hizo fine-tuning (puedes ajustar esos parámetros pasando los argumentos `max_seq_len` y `stride` al llamar el pipeline). Por lo tanto, usaremos esos parámetros al tokenizar. También agregaremos relleno (`padding`) (para tener muestras del mismo largo, para que podamos construir los tensores) como también pedir los offsets:

```py
inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
```

Esos `inputs` contendrán los IDs de entrada y las máscaras de atención (attention masks) que el modelo espera, así como los offsets y el `overflow_to_sample_mapping` que hablamos antes. Dado que esos dos no son parámetros usados por el modelo, los sacaremos de los `inputs` (y no guardaremos el mapa, ya que no es útil acá) antes de convertirlo en un tensor:


**PyTorch:**

```py
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)
```

```python out
torch.Size([2, 384])
```

**TensorFlow/Keras:**

```py
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("tf")
print(inputs["input_ids"].shape)
```

```python out
(2, 384)
```


Nuestro contexto largo fue dividido en dos, lo que significa que después de pasar por nuestro modelo, tendremos 2 sets de logits de inicio y término:

```py
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
```


**PyTorch:**

```python out
torch.Size([2, 384]) torch.Size([2, 384])
```

**TensorFlow/Keras:**

```python out
(2, 384) (2, 384)
```


Al igual que antes, primero enmascaramos los tokens que no son parte del contexto antes de aplicar softmax. También enmascaramos todos los tokens de de relleno (`padding`) (de acuerdo a la máscara de atención (attention masks)):


**PyTorch:**

```py
sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

start_logits[mask] = -10000
end_logits[mask] = -10000
```

**TensorFlow/Keras:**

```py
sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = tf.math.logical_or(tf.constant(mask)[None], inputs["attention_mask"] == 0)

start_logits = tf.where(mask, -10000, start_logits)
end_logits = tf.where(mask, -10000, end_logits)
```


Luego podemos usar la función softmax para convertir nuestros logits en probabilidades:


**PyTorch:**

```py
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)
```

**TensorFlow/Keras:**

```py
start_probabilities = tf.math.softmax(start_logits, axis=-1).numpy()
end_probabilities = tf.math.softmax(end_logits, axis=-1).numpy()
```


El siguiente paso es similar a lo que hicimos para el contexto pequeño, pero lo repetimos para cada uno de nuestros dos trozos. Le atribuímos un puntaje a todas las posibles respuestas, para luego tomar la respuesta con el mejor puntaje:


**PyTorch:**

```py
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)
```

**TensorFlow/Keras:**

```py
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = np.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)
```


```python out
[(0, 18, 0.33867), (173, 184, 0.97149)]
```

Estos dos candidatos corresponden a las mejores respuestas que el modelo fue capaz de encontrar en cada trozo. El modelo está mucho más confiado de que la respuesta correcta está en la segunda parte (¡lo que es una buena señal!). Ahora sólo tenemos que mapear dichos tokens a los caracteres en el contexto (sólo necesitamos mapear la segunda para obtener nuestra respuesta, pero es interesante ver que el modelo ha elegido en el primer trozo).

> [!TIP]
> ✏️ **Inténtalo!** Adapta el código de arriba pra retornar los puntajes de las 5 respuestas más probables (en total, no por trozo).

Los `offsets` que tomamos antes es en realidad una lista de offsets, con una lista por trozo de texto:

```py
for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
```

```python out
{'answer': '\n🤗 Transformers: State of the Art NLP', 'start': 0, 'end': 37, 'score': 0.33867}
{'answer': 'Jax, PyTorch and TensorFlow', 'start': 1892, 'end': 1919, 'score': 0.97149}
```

Si ignoramos el primer resultado, obtenemos el mismo resultado que nuestro pipeline para el contexto largo -- bien!

> [!TIP]
> ✏️ **Inténtalo!** Usa los mejores puntajes que calculaste antes para mostrar las 5 respuestas más probables. Para revisar tus resultados, regresa al primer pipeline y agrega `top_k=5` al llamarlo.

Esto concluye nuestra profundización en las capacidades de los tokenizadores. Pondremos todo esto en práctica de nuevo en el siguiente capítulo, cuando te mostremos cómo hacer fine-tuning a un modelo en una variedad de tareas comunes de PLN.


---

# Normalización y pre-tokenización[[normalization-and-pre-tokenization]]


Antes de sumergirnos más profundamente en los tres algoritmos más comunes de tokenización usados con los modelos transformers (Byte-Pair Encoding [BPE], WordPiece, and Unigram), primero miraremos el preprocesamiento que cada tokenizador aplica al texto. Acá una descripción general de los pasos en el pipeline de tokenización:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg" alt="The tokenization pipeline.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline-dark.svg" alt="The tokenization pipeline.">
</div>

Antes de dividir un texto en subtokens (de acuerdo a su modelo), el tokenizador realiza dos pasos: _normalización_ y _pre-tokenización_.

## Normalización[[normalization]]


**Video:** [Ver en YouTube](https://youtu.be/4IIC2jI9CaU)


El paso de normalización involucra una limpieza general, como la remoción de espacios en blanco innecesario, transformar a minúsculas, y/o remoción de acentos. Si estás familiarizado con [Normalización Unicode](http://www.unicode.org/reports/tr15/) (como NFC o NFKC), esto es algo que el tokenizador también puede aplicar.

Los tokenizadores de la librería 🤗 Transformers tienen un atributo llamado `backend_tokenizer` que provee acceso al tokenizador subyacente de la librería 🤗 Tokenizers:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
```

```python out
<class 'tokenizers.Tokenizer'>
```

El atributo `normalizer` del objeto `tokenizer` tiene un método `normalize_str()` que puede puedes usar para ver cómo la normalización se realiza:

```py
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

```python out
'hello how are u?'
```

En este ejemplo, dado que elegimos el punto de control (checkpoint) `bert-base-uncased`, la normalización aplicó transformación a minúsculas y remoción de acentos. 

> [!TIP]
> ✏️ **Inténtalo!** Carga un tokenizador desde el punto de control (checkpoint)`bert-base-cased` y pásale el mismo ejemplo. Cuáles son las principales diferencias que puedes ver entre las versiones cased y uncased de los tokenizadores?

## Pre-tokenización[[pre-tokenization]]


**Video:** [Ver en YouTube](https://youtu.be/grlLV8AIXug)


Como veremos en las siguientes secciones, un tokenizador no puede ser entrenado en un texto tal como viene así nada más. En vez de eso, primero necesitamos separar los textos en entidades más pequeñas, como palabras. Ahí es donde el paso de pre-tokenización entra en juego. Como vimos en el [Capítulo 2](/course/chapter2), un tokenizador basado en palabras (word-based) puede dividir el texto en palabras separando en espacios en blanco y puntuación. Esas palabras serán las fronteras de los subtokens que el tokenizador aprende durante su entrenamiento. 

Para ver qué tan rápido un tokenizador rápido (fast tokenizer) realiza la pre-tokenización, podemos usar el método `pre_tokenize_str()` del atributo `pre_tokenizer` del objeto `tokenizer`:

```py
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python out
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```

Notar como el tokenizador ya lleva registro de los offsets, el cual nos entrega el mapeo de offsets que usamos en la sección anterior. Acá el tokenizador ignora los dos espacios y los reemplaza con uno sólo, pero el offset salta entre `are` y `you` para tomar eso en cuenta. 

Dado que estamos usando un tokenizador BERT, la pre-tokenización involucra separar en espacios en blanco y puntuación. Otros tokenizadores pueden tener distintas reglas para esta etapa. Por ejemplo, si usamor el tokenizador de GPT-2:

```py
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

dividirá en espacios en blanco y puntuación también, pero mantendrá los espacios y los reemplazará con el símbolo `Ġ`, permitiendo recobrar los espacios originales en el caso de decodificar los tokens:

```python out
[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
 ('?', (19, 20))]
```

También notar que a diferencia del tokenizador BERT, este tokenizador no ignora los espacios dobles. 

Para el último ejemplo, tenemos que mirar el tokenizador T5, el cuál está basado en el algoritmo SentencePiece:

```py
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python out
[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```

Al igual que el tokenizador GPT-2, este mantiene los espacios y los reemplaza con un token específico (`_`), pero el tokenizador T5 sólo divide en espacios en blanco, no en puntuación. También notar que agrego un espacio por defecto al inicio de la oración (antes de `Hello`) e ignoró el doble espacio entre `are` y `you`.

Ahora que hemos visto un poco de cómo los diferentes tokenizadores procesan texto, podemos empezar a explorar los algoritmos subyacentes propiamente tal. Comenzaremos con una mirada rápida al ampliamente aplicable SentencePiece; luego, a lo largo de las 3 secciones siguientes examinaremos cómo los tres principales algoritmos usados para el trabajo de tokenización por subpalabra (subword tokenization).

## SentencePiece[[sentencepiece]]

[SentencePiece](https://github.com/google/sentencepiece) es un algoritmo para el preprocesamiento de texto que puedes usar con cualquiera de los modelos que veremos en las siguientes tres secciones. Éste considere el texto como una secuencia de caractéres Unicode, y reemplaza los especios con un caracter especial, `_`. Usado en conjunto con el algoritmo Unigram (ver [Sección 7](/course/chapter7/7)), ni siquiera requiere un paso de pre-tokenización, lo cual es muy útil para lenguajes donde el caracter de espacio no es usado (como el Chino o el Japonés).

La otra característica principal de SentencePiece es la *tokenización reversible* (tokenización reversible): dado que no hay tratamiento especial de los espacios, decodificar los tokens se hace simplemente concatenandolos y reemplazando los `_`s con espacios -- esto resulta en el texto normalizado. Como vimos antes, el tokenizador BERT remueve los espacios repetidos, por lo que su tokenización no es reversible. 

## Descripción General del Algoritmo[[algorithm-overview]]

En las siguientes secciones, profundizaremos en los tres principales algoritmos de tokenización por subpalabra (subword tokenization): BPE (usado por GPT-2 y otros), WordPiece (usado por ejemplo por BERT), y Unigram (usado por T5 y otros). Antes de comenzar, aquí una rápida descripción general de cómo funciona cada uno de ellos. No dudes en regresar a esta tabla luego de leer cada una de las siguientes secciones si no te hace sentido aún. 


Model | BPE | WordPiece | Unigram
:----:|:---:|:---------:|:------:
Entrenamiento | Comienza a partir de un pequeño vocabulario y aprende reglas para fusionar tokens |  Comienza a partir de un pequeño vocabulario y aprende reglas para fusionar tokens | Comienza de un gran vocabulario y aprende reglas para remover tokens
Etapa de Entrenamiento | Fusiona los tokens correspondiente a los pares más comunes | Fusiona los tokens correspondientes al par con el mejor puntaje basado en la frecuencia del par, privilegiando  pares donde cada token individual es menos frecuente | Remueve todos los tokens en el vocabulario que minimizarán la función de pérdida (loss) calculado en el corpus completo.
Aprende | Reglas de fusión y un vocabulario | Sólo un vocabulario | Un vocabulario con puntaje para cada token
Codificación | Separa una palabra en caracteres y aplica las fusiones aprendidas durante el entrenamiento | Encuentra la subpalabra más larga comenzando del inicio que está en el vocabulario, luego hace lo mismo para el resto de las palabras | Encuentra la separación en tokens más probable, usando los puntajes aprendidos durante el entrenamiento

Ahora profundicemos en BPE!

---

#  Tokenización por Codificación Byte-Pair[[byte-pair-encoding-tokenization]]


La codificación por pares de byte (Byte-Pair Encoding (BPE)) fue inicialmente desarrollado como un algoritmo para comprimir textos, y luego fue usado por OpenAI para la tokenización al momento de pre-entrenar el modelo GPT. Es usado por un montón de modelos Transformers, incluyendo GPT, GPT-2, RoBERTa, BART, y DeBERTa.


**Video:** [Ver en YouTube](https://youtu.be/HEikzVL-lZU)


> [!TIP]
> 💡 Esta sección cubre BPE en produndidad, yendo tan lejos como para mostrar una implementación completa. Puedes saltarte hasta el final si sólo quieres una descripción general del algoritmo de tokenización.

## Algoritmo de Entrenamiento[[training-algorithm]]

El entrenamiento de BPE comienza calculando el conjunto de palabras únicas usada en el corpus (después de completar las etapas de normalización y pre-tokenización), para luego contruir el vocabulario tomando todos los símbolos usados para escribir esas palabras. Como un ejemplo muy simple, digamos que nuestros corpus usa estas cinco palabras:


```
"hug", "pug", "pun", "bun", "hugs"
```

El vocabulario vase entonces será `["b", "g", "h", "n", "p", "s", "u"]`. Para casos reales, el vocabulario base contendrá todos los caracteres ASCII, al menos, y probablemente algunos caracteres Unicode también. Si un ejemplo que estás tokenizando usa un caracter que no está en el corpus de entrenamiento, ese caracter será convertido al token "desconocido". Esa es una razón por la cual muchos modelos de NLP son muy malos analizando contenido con emojis.

> [!TIP]
> Los tokenizadores de GPT-2 y RoBERTa (que son bastante similares) tienen una manera bien inteligente de lidiar con esto: ellos no miran a las palabras como si estuvieran escritas con caracteres Unicode, sino con bytes. De esa manera el vocabulario base tiene un tamaño pequeño (256), pero cada caracter que te puedas imaginar estará incluido y no terminará convertido en el token "desconocido". Este truco se llama *byte-level BPE*.

Luego de obtener el vocabulario base, agregamos nuevos tokens hasta que el tamaño deseado del vocabulario se alcance por medio de aprender *fusiones* (merges), las cuales son reglas para fusionar dos elementos del vocabulario existente en uno nuevo. Por lo que al inicio de estas fusiones crearemos tokens con dos caracteres, y luego, a medida que el entrenamiento avance, subpalabras más largas.

En cualquier etapa durante el entrenamiento del tokenizador, el algoritmo BPE buscará pos los pares más frecuentes de los tokens existentes (por "par", acá nos referimos a dos tokens consecutivos en una palabra). El par más frecuente es el que será fusionado, y enjuagamos y repetimos para la siguiente etapa. 

Volviedo a nuestro ejemplo previo, asumamos que las palabras tenían las siguientes frecuencias:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

lo que significa que `"hug"` estuvo presente 10 veces en el corpus, `"pug"` 5 veces, `"pun"` 12 veces, `"bun"` 4 veces, and `"hugs"` 5 veces. Empezamos el entrenamiento separando cada palabra en caracteres (los que formaron nuestro vocabulario inicial) para que podamos ver cada palabra como una lista de tokens:

```
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

Luego miramos los pares. El par `("h", "u")` está presente en las palabras `"hug"` y `"hugs"`, 15 veces en el total del corpus. No es el par más frecuente: ese honor le corresponde a `("u", "g")`, el cual está presente en `"hug"`, `"pug"`, y `"hugs"`, para un gran total de 20 veces en el vocabulario. 

Por lo tanto, la primera regla de fusión aprendida por el tokenizador es `("u", "g") -> "ug"`, lo que significa que `"ug"` será agregado al vocabulario, y el par debería ser fusionado en todas las palabras del corpus. Al final de esta etapa, el vocabulario se ve así:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

Ahora tenemos algunos pares que resultan en un token más largo de dos caracteres: por ejemplo el par `("h", "ug")` (presente 15 veces en el corpus). Sin embargo, el par más frecuente en este punto is `("u", "n")`, presente 16 veces en el corpus, por lo que la segunda regla de fusión aprendida es `("u", "n") -> "un"`. Agregando esto y fusionando todas las ocurrencias existentes nos lleva a:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

Ahora el par más frecuente es `("h", "ug")`, por lo que aprendemos que la regla de fusión es `("h", "ug") -> "hug"`, lo cual nos da tuestro primer token de tres letras. Luego de la fusión el corpus se ve así:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
Corpus: ("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

Y continuamos así hasta que alcancemos el tamaño deseado del vocabulario.

> [!TIP]
> ✏️ **Ahora es tu turno!** Cuál crees que será la siguiente regla de fusión?

## Algoritmo de Tokenización[[tokenization-algorithm]]

La tokenización sigue el proceso de entrenamiento de cerca, en el sentido que nuevos inputs son tokenizados aplicando los siguientes pasos:

1. Normalización
2. Pre-tokenización
3. Separar las palabras en caracteres individuales
4. Aplicar las reglas de fusión aprendidas en orden en dichas separaciones.

Tomemos el ejemplo que usamos durante el entrenamiento, con las tres reglas de fusión aprendidas:

```
("u", "g") -> "ug"
("u", "n") -> "un"
("h", "ug") -> "hug"
```
La palabra `"bug"` será tokenizada como `["b", "ug"]`. En cambio, `"mug"`, será tokenizado como `["[UNK]", "ug"]` dado que la letra `"m"` no fue parte del vocabulario base. De la misma manera, la palabra `"thug"` será tokenizada como `["[UNK]", "hug"]`: la letra `"t"` no está en el vocabulario base, y aplicando las reglas de fusión resulta primero la fusión de `"u"` y `"g"` y luego de `"hu"` and `"g"`.

> [!TIP]
> ✏️ **Ahora es tu turno!** ¿Cómo crees será tokenizada la palabra `"unhug"`?

## Implementando BPE[[implementing-bpe]]

Ahora echemos un vistazo a una implementación el algoritmo BPE. Esta no será una versión optimizada que puedes usar en corpus grande; sólo queremos mostrar el código para que puedas entender el algoritmo un poquito mejor. 

Primero necesitamos un corpus, así que creemos uno simple con algunas oraciones:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

A continuación, necesitamos pre-tokenizar el corpus en palabras. Dado que estamos replicando un tokenizador BPE (como GPT-2), usaremos el tokenizdor `gpt2` para la pre-tokenización:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Luego calculamos las frecuencias de cada palabra en el corpues mientras hacemos la pre-tokenización:

```python
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
```

```python out
defaultdict(int, {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1,
    'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1,
    'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1,
    'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1})
```

El siguiente paso es calcualar el vocabulario base, formado por todos los caracteres usados en el corpus:

```python
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)
```

```python out
[ ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
  't', 'u', 'v', 'w', 'y', 'z', 'Ġ']
```

También agregamos el token especial usado por el modelo al inicio de ese vocabulario. En el caso de GPT-2, el único token especial es `"<|endoftext|>"`:

```python
vocab = ["<|endoftext|>"] + alphabet.copy()
```

Ahora necesitamos separar cada palabra en caracteres individuales, para poder comenzar el entrenamiento:

```python
splits = {word: [c for c in word] for word in word_freqs.keys()}
```

Ahora estamos listos para el entrenamiento, escribamos una función que calcule la frecuencia de cada par. Necesitaremos usar esto en cada paso del entrenamiento:

```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
```

Ahora miremos una parte de ese diccionario después de las separaciones iniciales:

```python
pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break
```

```python out
('T', 'h'): 3
('h', 'i'): 3
('i', 's'): 5
('Ġ', 'i'): 2
('Ġ', 't'): 7
('t', 'h'): 3
```

Ahora, encontrar el par más frecuenta sólo toma un rápido ciclo:

```python
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
```

```python out
('Ġ', 't') 7
```

Por lo que la primera fusión a aprender es `('Ġ', 't') -> 'Ġt'`, y luego agregamos `'Ġt'` al vocabulario:

```python
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")
```

Para continuar, necesitamos aplicar la fusión en nuestro diccionario de divisiones (`splits` dictionary). Escribamos otra función para esto:

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

Y podemos echar un vistazo al resultado de nuestra primera fusión:

```py
splits = merge_pair("Ġ", "t", splits)
print(splits["Ġtrained"])
```

```python out
['Ġt', 'r', 'a', 'i', 'n', 'e', 'd']
```

Ahora tenemos todo lo que necesitamos para iterar hasta que aprendamos todas las fusiones que queramos. Apuntemos a un tamaño de vocabulario de 50:

```python
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```

Como resultado, hemos aprendido 19 reglas de fusión (el vocabulario inicial tenía un tamaño de 31 -- 30 caracteres del alfabeto, más el token especial):

```py
print(merges)
```

```python out
{('Ġ', 't'): 'Ġt', ('i', 's'): 'is', ('e', 'r'): 'er', ('Ġ', 'a'): 'Ġa', ('Ġt', 'o'): 'Ġto', ('e', 'n'): 'en',
 ('T', 'h'): 'Th', ('Th', 'is'): 'This', ('o', 'u'): 'ou', ('s', 'e'): 'se', ('Ġto', 'k'): 'Ġtok',
 ('Ġtok', 'en'): 'Ġtoken', ('n', 'd'): 'nd', ('Ġ', 'is'): 'Ġis', ('Ġt', 'h'): 'Ġth', ('Ġth', 'e'): 'Ġthe',
 ('i', 'n'): 'in', ('Ġa', 'b'): 'Ġab', ('Ġtoken', 'i'): 'Ġtokeni'}
```

And the vocabulary is composed of the special token, the initial alphabet, and all the results of the merges:

```py
print(vocab)
```

```python out
['<|endoftext|>', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'Ġ', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 'en', 'Th', 'This', 'ou', 'se',
 'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni']
```

> [!TIP]
> 💡 Usar `train_new_from_iterator()` en el mismo corpus no resultará en exactament el mismo vocabulario. Esto es porque cuando hay una elección del par más frecuente, seleccionamos el primero encontrado, mientras que la librería 🤗 Tokenizers selecciona el primero basado en sus IDs internos.

Para tokenizar un nuevo texto lo pre-tokenizamos, lo separamos, luego aplicamos todas las reglas de fusión aprendidas:

```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
```

Podemos intentar esto con cualquier texto compuesto de de caracteres del alfabeto:

```py
tokenize("This is not a token.")
```

```python out
['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']
```

> [!WARNING]
> ⚠️ Nuestra implementación arrojará un error si hay un caracter desconocido dado que no hicimos nada para manejarlos. GPT-2 en realidad no tiene un token desconocido (es imposible obtener un caracter desconocido cuando se usa byte-level BPE), pero esto podría ocurrir acá porque no incluímos todos los posibles bytes en el vocabulario inicial. Este aspectode BPE va más allá del alcance de está sección, por lo que dejaremos los detalles fuera.

Eso es todo para el algoritmo BPE! A continuación echaremos un vistazo a WordPiece.

---

# Tokenización WordPiece[[wordpiece-tokenization]]


WordPiece es el algoritmo de tokenización que Google desarrolló para pre-entrenar BERT. Ha sido reutilizado un varios modelos Transformers basados en BERT, tales como DistilBERT, MobileBERT, Funnel Transformers, y MPNET. Es muy similar a BPE en términos del entrenamiento, pero la tokenización se hace de distinta manera. 


**Video:** [Ver en YouTube](https://youtu.be/qpv6ms_t_1A)


> [!TIP]
> 💡 Esta sección cubre WordPiece en profundidad, yendo tan lejos como para mostrar una implementación completa. Puedes saltarte hasta el final si sólo quieres una descripción general del algoritmo de tokenización.

## Algoritmo de Entrenamiento[[training-algorithm]]

> [!WARNING]
> ⚠️ Google nunca liberó el código (open-sourced) su implementación del algoritmo de entrenamiento de WordPiece, por tanto lo que sigue es nuestra mejor suposición badado en la literatura publicada. Puede no ser 100% preciso.

Al igual que BPE, WordPiece comienza a partir de un pequeño vocabulario incluyendo los tokens especiales utilizados por el modelo y el alfabeto inicial. Dado que identifica subpalabras (subwords) agregando un prefijo (como `##` para el caso de BERT), cada palabra está inicialmente separada agregando dicho prefijo a todos los caracteres dentro de la palabra. Por lo que por ejemplo la palabra `"word"` queda separada así:

```
w ##o ##r ##d
```
Por lo tanto, el alfabeto inicial contiene todos los caracteres presentes al comienzo de una palabra y los caracteres presente dentro de una palabra precedida por el prefijo de WordPiece. 

Luego, de nuevo al igual que BPE, WordPiece aprende reglas de fusión. La principal diferencia es la forma que el par fusionado es seleccionado. Envex de seleccionar el par más frecuente, WordPiece calcula un puntaje para cada par, utilizando la siguiente formula:

$$\mathrm{score} = (\mathrm{freq\_of\_pair}) / (\mathrm{freq\_of\_first\_element} \times \mathrm{freq\_of\_second\_element})$$

Dividiendo por la frecuencia del par por el producto de las frecuencias de cada una de sus partes, el algoritmo prioriza la fusión de pares donde las partes individuales son menos frecuentes en el vocabulario. Por ejemplo, no fusionará necesariamente `("un", "##able")` incluso si ese par ocurre de manera muy frecuente en el vocabulario, porque los dos pares `"un"` y `"##able"` muy probablemente aparecerán en un montón de otras palabras y tendrán una alta frecuencia. En contraste con un par como `("hu", "##gging")` los cuales son probablemente menos frecuentes individualmente. 

Miremos el mismo vocabulario que usamos en el ejemplo de entrenamiento de BPE:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

Las separaciones acá serán:

```
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)
```

por lo que el vocabulario inicial será `["b", "h", "p", "##g", "##n", "##s", "##u"]` (si nos olvidamos de los tokens especiales por ahora). El par más frecuente es `("##u", "##g")` (presente 20 veces), pero la frecuencia individual de `"##u"` es muy alta, por lo que el puntaje no es el más alto (es 1 / 36). Todos los pares con `"##u"` en realidad tienen el mismo puntaje (1 / 36), por lo que el mejor puntaje va para el par `("##g", "##s")` -- el único sin `"##u"` -- 1 / 20, y la primera fusión aprendida es `("##g", "##s") -> ("##gs")`.

Notar que cuando fusionamos, removemos el `##` entre los dos tokens, por que agregamos `"##gs"` al vocabulario y aplicamos la fusión en las palabras del corpus:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)
```

En este punto, `"##u"` está en todos los posibles pares, por lo que todos terminan con el mismo puntaje. Digamos que en este caso, el primer par se fusiona, `("h", "##u") -> "hu"`. Esto nos lleva a:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"]
Corpus: ("hu" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```

Luego el siguiente mejor puntaje está compartido por `("hu", "##g")` y `("hu", "##gs")` (con 1/15, comparado con 1/21 para todos los otros pares), por lo que el primer par con el puntaje más alto se fusiona:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hug"]
Corpus: ("hug", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```

y continuamos como esto hasta que alcancemos el tamaño de vocabulario deseado.

> [!TIP]
> ✏️ **Ahora es tu turno!** Cuál será la siguiente regla de fusioń?

## Algoritmo de Tokenización[[tokenization-algorithm]]

La tokenización difiere en WordPiece y BPE en que WordPiece sólo guarda el vocabulario final, no las reglas de fusión aprendidas. Comenzando a partir de la palabra a tokenizar, WordPiece encuentra la subpalabra más larga que está en el vocabulario, luego la separa. Por ejemplo, su usamos el vocabulario aprendido en el ejemplo anterior, para la palabra `"hugs"` la subpalabra más larga comenzando desde el inicio que está dentro del vocabulario es `"hug"`, por lo que separamos ahí y obtenemos `["hug", "##s"]`. Luego continuamos con `"##s"`, el cuál está en el vocabulario, por lo que la tokenización de `"hugs"` es `["hug", "##s"]`.

Con BPE, habríamos aplicado las fusiones aprendidas en orden y tokenizado esto como `["hu", "##gs"]`, por lo que la codificación es diferente. 

Como otro ejemplo, veamos como la palabra `"bugs"` sería tokenizado. `"b"` es la subpalabra más larga comenzando del inicio de la palabra que está en el vocabulario, por lo que separamos ahí y obtenemos `["b", "##ugs"]`. Luego `"##u"` es la subpalabra más larga somenzando desde el inicio de `"##ugs"` que está en el vocabulario, por lo que separamos ahí y obtenemos `["b", "##u, "##gs"]`. Finalmente, `"##gs"` está en el vocabulario, por lo que esta última lista es la tokenización de `"bugs"`.

Cuando la tokenización llega a la etapa donde ya no es posible encontrar una subpalabra en el vocabulario, la palabra entera es tokenizada como desconocida -- Por ejemplo, `"mug"` sería tokenizada como `["[UNK]"]`, al igual que `"bum"` (incluso si podemos comenzar con `"b"` y `"##u"`, `"##m"` no está en el vocabulario, y la tokenización resultante será sólo `["[UNK]"]`, y no `["b", "##u", "[UNK]"]`). Este es otra diferencia con respecto a BPE, el cual sólo clasificaría los caracteres individuales que no están en el vocabulario como desconocido.

> [!TIP]
> ✏️ **Ahora es tu turno!** ¿Cómo se tokenizaría la palabra `"pugs"`?

## Implementando WordPiece[[implementing-wordpiece]]

Ahora echemos un vistazo a una implementación del algoritmo WordPiece. Al igual que BPE, este es sólo pedagócico y no podrás aplicar esto en corpus grande. 

Usaremos el mismo corpus que en el ejemplo de BPE:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

Primero, necesitamos pre-tokenizar el corpus en palabras. Dado que estamos replicando el tokenizador WordPiece (como BERT), usaremos el tokenizador `bert-base-cased` para la pre-tokenización:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Luego calculamos las frecuencias de cada palabra en el corpus mientras hacemos la pre-tokenización:

```python
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```

```python out
defaultdict(
    int, {'This': 3, 'is': 2, 'the': 1, 'Hugging': 1, 'Face': 1, 'Course': 1, '.': 4, 'chapter': 1, 'about': 1,
    'tokenization': 1, 'section': 1, 'shows': 1, 'several': 1, 'tokenizer': 1, 'algorithms': 1, 'Hopefully': 1,
    ',': 1, 'you': 1, 'will': 1, 'be': 1, 'able': 1, 'to': 1, 'understand': 1, 'how': 1, 'they': 1, 'are': 1,
    'trained': 1, 'and': 1, 'generate': 1, 'tokens': 1})
```

Como vimos antes, el alfabeto es el único conjunto compuesto de todas las primeras letras de las palabras, y todas las otras letras que aparecen con el prefijo `##`:

```python
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
alphabet

print(alphabet)
```

```python out
['##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
 '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u',
 'w', 'y']
```

También agregamos los tokens especiales usados por el modelo al inicio de ese vocabulario. En el caso de BERT, es la lista `["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]`:

```python
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
```

A continuación necesitamos separar cada palabra, con todas las letras que no tienen el prefijo `##`:

```python
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}
```

Ahora que estamos listos para el entrenamiento, escribamos una función que calcule el puntaje para cada par. Usaremos esto en cada etapa del entrenamiento:

```python
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores
```

Echemos un vistazo a parte de este diccionario luego de las separaciones iniciales:

```python
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break
```

```python out
('T', '##h'): 0.125
('##h', '##i'): 0.03409090909090909
('##i', '##s'): 0.02727272727272727
('i', '##s'): 0.1
('t', '##h'): 0.03571428571428571
('##h', '##e'): 0.011904761904761904
```

Ahora, encontrar el par con el mejor puntaje sólo toma un rápido ciclo:

```python
best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)
```

```python out
('a', '##b') 0.2
```

Por lo que la primera fusión a aprender es `('a', '##b') -> 'ab'`, y agregamos `'ab'` al vocabulario:

```python
vocab.append("ab")
```

Para continuar, necesitamos aplicar esa fusión en nuestro diccionario de separaciones (`splits` dictionary). Escribamos otra función para esto: 

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

Y podemos mirar el resultado de la primera fusión:

```py
splits = merge_pair("a", "##b", splits)
splits["about"]
```

```python out
['ab', '##o', '##u', '##t']
```

Ahora tenemos todos los que necesitamos para iterar hasta haber aprendido todas las fusiones que queramos. Apuntemos a un tamaño de vocabulario de 70:

```python
vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
```

Luego podemos ver el vocabulario generado:

```py
print(vocab)
```

```python out
['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k',
 '##l', '##m', '##n', '##o', '##p', '##r', '##s', '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H',
 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully',
 'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat',
 '##ut']
```

Como podemos ver, comparado con BPE, este tokenizador aprende partes de palabras como tokens un poco más rápido.

> [!TIP]
> 💡 Usar `train_new_from_iterator()` en el mismo corpus no resultará en exactamente el mismo vocabulario. Esto porque la librería 🤗 Tokenizers no implementa WordPiece para el entrenamiento (dado que no estamos completamente seguros de su funcionamiento interno), en vez de eso utiliza BPE.

Para tokenizar un nuevo texto, lo pre-tokenizamos, lo separamos, y luego aplicamos el algoritmo de tokenización para cada palabra. Es decir, miramos la subpalabra más grande comenzando al inicio de la primera palabra y la separamos, luego repetimos el proceso en la segunda parte, y así pará el resto de dicha palabra y de las siguientes palabras en el texto:

```python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
```

Probémoslo en una palabra que esté en el vocabulario, y en otra que no esté:

```python
print(encode_word("Hugging"))
print(encode_word("HOgging"))
```

```python out
['Hugg', '##i', '##n', '##g']
['[UNK]']
```

Ahora, escribamos una función que tokenize un texto:

```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])
```

Podemos probar en cualquier texto:

```python
tokenize("This is the Hugging Face course!")
```

```python out
['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'c', '##o', '##u', '##r', '##s',
 '##e', '[UNK]']
```

Eso es todo para el algoritmo WordPiece! Ahora echemos un visto a Unigram.


---

# Tokenización Unigram[[unigram-tokenization]]


El algoritmo de Unigram es a menudo utilizado en SetencePiece, el cual es el algoritmo de tokenización usado por modelos como AlBERT, T5, mBART, Big Bird y XLNet.


**Video:** [Ver en YouTube](https://youtu.be/TGZfZVuF9Yc)


> [!TIP]
> 💡 Esta sección cubre Unigram en profundidad, yendo tan lejos como para mostrar una implementación completa. Puedes saltarte hasta el final si sólo quieres una descripción general del algoritmo de tokenización.

## Algoritmo de Entrenamiento[[training-algorithm]]

Comparado con BPE y WordPiece, Unigram funciona en la otra dirección: comienza desde un gran vocabulario y remueve tokens hasta que alcanza el tamaño deseado del vocabulario.. Hay varias opciones para construir el vocabulario base: podemos tomar los substrings más comunes en palabras pre-tokenizadas, por ejemplo, o aplicar BPE en el corpus inicial con un tamaño de vocabulario grande. 

En cada paso del entrenamiento, el algoritmo de Unigram calcula la pérdida (`loss`)sobre el corpus dado el vocabulario actual. Entonces para cada símbolo en el vocabulario, el algoritmo calcula cuánto incremetaría el la pérdida (`loss`) total si el símbolo se remueve, y busca por los símbolos que lo incrementarían lo menos posible. Esos símbolos tienen un efecto más bajo en la pérdida sobre el corpus, por lo que en un sentido son "menos necesarios" y son los mejores candidatos para ser removidos. 

Esto es una operación bastante costosa, por lo que no removemos un sólo símbolo asociato con el incremento en la pérdida (`loss`) más baja, sino que \\(p\\) (\\(p\\) es un parámetro que puedes controlar, usualmente 10 o 20) porciento de los símbolos asociados con el incremento más bajo de la pérdida. Este proceso es repetido hasta que el vocabulario ha alcanzado el tamaño deseado. 

Nota que nunca removemos los caracteres base, para asegurarnos que cada palabra pueda ser tokenizada. 

Hora, esto es todavía un poco vago: la parte principal del algoritmo es calcular una pérdida (`loss`) sobre el corpus, y ver como cambia cuando removemos algunos tokens desde el vocabulario, pero no hemos explicado como hacer esto aún. Este paso se basa en el algoritmo de tokenización de un modelo Unigram, por lo que profundizaremos en esto a continuación.

Usaremos el corpus de los ejemplos previos:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

y para este ejemplo, tomaremos todos los substrings strictos para el vocabulario inicial. 

```
["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
```

## Algoritmo de Tokenización[[tokenization-algorithm]]

Un modelo Unigram es un tipo de modelo de lenguaje que considera cada token como independiente de los tokens antes que él. Es el modelo de lenguaje más simple, en el sentido de que la probabilidad de que el token X dado el contexto previo es sólo la probabilidad del token X. Por lo que, si usamos un modelo de Lenguaje Unigram para generar texto, siempre predeciríamos el token más común.

La probabilidad de un token dado es su frecuencia (el número de veces en el cual lo encontramos) en el corpus original, dividido por la suma de todas las frecuencias de todos los tokens en el vocabulario (para asegurarnos que las probabilidad sumen 1). Por ejemplo, `"ug"` está presente en `"hug"`, `"pug"`, y `"hugs"`, por lo que tiene una frecuencia de 20 en nuestro corpus. 

Acá están las frecuencias de todas las posibles subpalabras en el vocabulario:

```
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)
```

Por lo que, la suma de todas las frecuencias es 210, y la probabilidad de la subpalabra `"ug"` es por lo tanto 20/210.

> [!TIP]
> ✏️ **Ahora es tu turno!** Escribe el código para calcular las frecuencias de arriba y chequea que los resultados mostrados son correctos, como también la suma total.

Ahora, para tokenizar una palabra dada, miramos todas las posibles segmentaciones en tokens y calculamos la probabilidad de cada uno de acuerdo al modelo Unigram. Dado que todos los tokens se consideran como independientes, esta probabilidad es sólo el producto de la probabilidad de cada token. Por ejemplo, la tokenización `["p", "u", "g"]` de `"pug"` tiene como probabilidad: 

$$P([``p", ``u", ``g"]) = P(``p") \times P(``u") \times P(``g") = \frac{5}{210} \times \frac{36}{210} \times \frac{20}{210} = 0.000389$$

Comparativamente, la tokenización `["pu", "g"]` tiene como probabilidad:

$$P([``pu", ``g"]) = P(``pu") \times P(``g") = \frac{5}{210} \times \frac{20}{210} = 0.0022676$$

por lo que es un poco más probable. En general, las tokenizaciones con el menor número de tokens posibles tendrán la probabilidad más alta (debido a la división por 210 repetida para cada token), lo cual corresponde a lo que queremos intuitivamente: separar una palabra en el menor número de tokens posibles. 

La tokenización de una palabra con el modelo Unigram es entonces la tokenización con la probabilidad más alta. Acá están las probabilidades para el ejemplo de `"pug"` que obtendríamos para cada posible segmentación:

```
["p", "u", "g"] : 0.000389
["p", "ug"] : 0.0022676
["pu", "g"] : 0.0022676
```

Por lo que, `"pug"` sería tokenizado como `["p", "ug"]` o `["pu", "g"]`, dependiendo de cual de esas segmentaciones e encuentre primero (notar que en un corpus grande, casos equivalentes como este serán raros).

En este caso, fue fácil encontrar todas las posibles segmentaciones y calcular sus probabilidades, pero en general, va a ser un poco más difícil. Hay un algoritmo clásico usado para esto, llamado el *Algoritmo de Viterbi* (*Viterbi algorithm*). Esencialmente, podemos construir un grafo para detectar las posibles segmentaciones de una palabra dada diciendo que existe una rama que va desde el caracter _a_ hasta el caracter _b_ si la subpalabra de _a_ hasta _b_ está en el vocabulario, y se atribuye a esa rama la probabilidad de la subpalabra. 

Para encontrar el camino en dicho grafo que va a tener el mejor puntaje el Algoritmo de Viterbi determina, por cada posición en la palabra, la segmentacion con el mejor puntaje que termina en esa posición. Dado que vamos desde el inicio al final, el mejor puntaje puede ser encontrado iterando a través de todas las subpalabras que terminan en la posición actual y luego usando el mejor puntaje de tokenización desde la posición en que esta palabra comienza. Luego sólo tenemos que desenrollar el camino tomado para llegar al final. 

Echemos un vistazo a un ejemplo usando nuestro vocabulario y la palabra `"unhug"`. Para cada posición, las subpalabras con el mejor puntaje terminando ahí son las siguientes:

```
Character 0 (u): "u" (score 0.171429)
Character 1 (n): "un" (score 0.076191)
Character 2 (h): "un" "h" (score 0.005442)
Character 3 (u): "un" "hu" (score 0.005442)
Character 4 (g): "un" "hug" (score 0.005442)
```

Por lo tanto, `"unhug"` se tokenizaría como `["un", "hug"]`.

> [!TIP]
> ✏️ **Ahora es tu turno!** Determina la tokenización de la palabra `"huggun"`, y su puntaje

## De vuelta al entrenamiento[[back-to-training]]

Ahora que hemos visto cómo funciona la tokenización, podemos ir un poco más profundo en la pérdida (`loss`) usada durante el entrenamiento. En cualquier etapa, esta pérdida (`loss`) es calculada tokenizando cualquier palabra en el corpus, usando el vocabulario actual y el modelo Unigram determinado por las frecuencias de cada token en el corpus (como se vió antes).

Cada palabra en el corpus tiene un puntaje, y la pérdida (`loss`) es la log verosimilitud negativa (negative log likelihood) de estos puntajes -- es decir, la suma por todas las palabras en el corpus de todos los `-log(P(word))`.

Volvamos a nuestro ejemplo con el siguiente corpus:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

La tokenización de cada palabra con sus respectivos puntajes es:

```
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)
```

Por lo que la loss es:

```
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8
```

Ahora necesitamos calcular cómo remover cada token afecta a la pérdida (`loss`). Esto es bastante tedioso, por lo que lo haremos sólo para dos tokens avá y nos ahorraremos el proceso entero para cuando tengamos código que nos ayude. En este (muy) particular caso, teníamos dos tokenizaciones equivalentes de todas las palabras: como vimos antes, por ejemplo, `"pug"` podría ser tokenizado como `["p", "ug"]` con el mismo puntaje. Por lo tanto, removiendo el token `"pu"` del vocabulario nos dará la misma pérdida.

Por otro lado, remover, `"hug"` hará nuestra pérdida peor, porque la tokenización de `"hug"` y `"hugs"` se convertirá en:

```
"hug": ["hu", "g"] (score 0.006802)
"hugs": ["hu", "gs"] (score 0.001701)
```

Estos cambios causarán que la pérdida aumenta en:

```
- 10 * (-log(0.071428)) + 10 * (-log(0.006802)) = 23.5
```

Por lo tanto, el token `"pu"` será probablemente removido del vocabulario, pero `"hug"`.

## Implementando Unigram[[implementing-unigram]]

Ahora, implementemos todo lo que hemos visto hasta ahora en código. Al igual que BPE y WordPiece, esta es una implementación no tan eficiente del algoritmo Unigram (de hecho, todo lo contrario), pero debería ayudar a entenderla un poco mejor. 

Usaremos el mismo corpus que antes como nuestro ejemplo:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

Esta vez, usaremos `xlnet-base-cased` como nuestro modelo:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
```

Al igual que BPE y WordPiece, comenzamos contando el número de ocurrencias para cada palabra en el corpus:

```python
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```

Luego, necesitamos inicializar nuestro vocabulario a algo más grande que el tamaño de vocabulario que querremos al final. Tenemos que incluir, todos los caracteres básicos (de otra manera no seremos capaces de tokenizar cada palabra), pero para los substrings más grandes mantendremos sólos los más comunes, de manera que los ordenemos por frecuencia:

```python
char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# Sort subwords by frequency
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]
```

```python out
[('▁t', 7), ('is', 5), ('er', 5), ('▁a', 5), ('▁to', 4), ('to', 4), ('en', 4), ('▁T', 3), ('▁Th', 3), ('▁Thi', 3)]
```

Agrupamos los caracteres con las mejores subpalabras para llegar a un vocabulario inicial de 300:

```python
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}
```

> [!TIP]
> 💡 SentencePiece usa un algoritmo más eficiente llamado Enhanced Suffix Array (ESA) para crear el vocabulario inicial.

A continuación, calculamos la suma de todas las frecuencias, para convertir las frecuencias en probabilidades. Para nuestro modelo, almacenaremos los logaritmos de las probabilidades, porque es numericamente más estable sumar logaritmos que multiplicar números pequeños, y esto simplificará el cálculo de la pérdida (`loss`) del modelo:

```python
from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```

Ahora, la función es la que tokeniza palabras usando el algoritmo de Viterbi. Como vimos antes, el algoritmo calcula la mejor segmentación de cada substring de la palabra, la cual almacenará en una variable llamada `best_segmentations`. Almacenaremos un diccionario por posición en la palabra (desde 0 hasta su largo total), con dos claves: el índice de inicio del último token en la mejor segmentación, y el puntaje de la mejor segmentación. Con el índice del inicio del último token, seremos capaces de recuperar la segmentación total una vez que la lista esté completamente poblada. 

Poblar la lista se hace con dos ciclos: el ciclo principal recorre cada posición de inicio, y el segundo loop, prueba todos los substrings comenaando en esa posición. Si el substring está en el vocabulario, tenemos una nueva segmentación de la palabra hasta esa posición final, la cual comparamos con lo que está en `best_segmentations`.

Una vez que el ciclo principal se termina, empezamos desde el final y saltamos de una posición de inicio hasta la siguiente, guardando los tokens a medida que avanzamos, hasta alcanzar el inicio de la palabra:

```python
def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
```

Ya podemos probar nuestro modelo inicial en algunas palabras:

```python
print(encode_word("Hopefully", model))
print(encode_word("This", model))
```

```python out
(['H', 'o', 'p', 'e', 'f', 'u', 'll', 'y'], 41.5157494601402)
(['This'], 6.288267030694535)
```

Ahora es fácil calcular la pérdida (`loss`) del modelo en el corpus!

```python
def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss
```

Podemos chequear que funciona en el modelo que tenemos:

```python
compute_loss(model)
```

```python out
413.10377642940875
```

Calcular los puntajes para cada token no es tan difícil tampoco; sólo tenemos que calcular la pérdida para los modelos obtenidos al eliminar cada token:

```python
import copy


def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores
```

Podemos probarlo en token dado: 

```python
scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])
```

Dado que `"ll"` se usa en la tokenización de `"Hopefully"`, y removerlo nos hará probablemente usar el token `"l"` dos veces, esperamos que tendrá una pérdida positiva. `"his"` es sólo usado dentro de la palabra `"This"`, lo cuál es tokenizado como sí mismo, por lo que esperamos que tenga pérdida cero. Acá están los resultados:

```python out
6.376412403623874
0.0
```

> [!TIP]
> 💡 Este acercamiento es muy ineficiente, por lo que SentencePiece usa una aproximación de la pérdida del modelo sin el token X: en vez de comenzar desde cero, sólo reemplaza el token X por su segmentación en el vocabulario que queda. De esta manera, todos los puntajes se pueden calcular de una sóla vez al mismo tiempo que la pérdida del modelo.

Con todo esto en su lugar, lo último que necesitamos hacer es agregar los tokens especiales usados por el modelo al vocabulario, e iterar hasta haber podado suficientes tokens de nuestro vocabulario hasta alcanzar el tamaño deseado:

```python
percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```

Luego, para tokenizar algo de texto, sólo necesitamos aplicar la pre-tokenización y luego usar nuestra función `encode_word()`:

```python
def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)
```

```python out
['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']
```

Eso es todo para Unigram! Ojalá a esta altura te sientas como un experto en todos los aspectos de los tokenizadores. En la siguiente sección, ahondaremos en las unidades básicas de la librería 🤗 Tokenizers, y te mostraremos cómo puedes usarlo para construir tu propio tokenizador.

---

# Construir un tokenizador, bloque por bloque[[building-a-tokenizer-block-by-block]]


Como hemos visto en las secciones previas, la tokenización está compuesta de varias etapas:

- Normalización (cualquier limpieza del texto que se considere necesaria, tales como remover espacios o acentos, normalización Unicode, etc.)
- Pre-tokenización (separar la entrada en palabras)
- Pasar las entradas (inputs) por el modelo (usar las palabras pre-tokenizadas para producir una secuencia de tokens)
- Post-procesamiento (agregar tokens especiales del tokenizador, generando la máscara de atención (attention mask) y los IDs de tipo de token)

Como recordatorio, acá hay otro vistazo al proceso en totalidad:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg" alt="The tokenization pipeline.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline-dark.svg" alt="The tokenization pipeline.">
</div>

La librería 🤗 Tokenizers ha sido construida para proveer varias opciones para cada una de esas etapas, las cuales se pueden mezclar y combinar. En esta sección veremos cómo podemos construir un tokenizador desde cero, opuesto al entrenamiento de un nuevo tokenizador a partir de uno existente como hicimos en la [Sección 2](/course/chapter6/2). Después de esto, serás capaz de construir cualquier tipo de tokenizador que puedas imaginar!


**Video:** [Ver en YouTube](https://youtu.be/MR8tZm5ViWU)


De manera más precisa, la librería está construida a partir de una clase central `Tokenizer` con las unidades más básica reagrupadas en susbmódulos: 

- `normalizers` contiene todos los posibles tipos de `Normalizer` que puedes usar (la lista completa [aquí](https://huggingface.co/docs/tokenizers/api/normalizers)).
- `pre_tokenizers` contiene todos los posibles tipos de `PreTokenizer` que puedes usar (la lista completa [aquí](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)).
- `models` contiene los distintos tipos de `Model` que puedes usar, como `BPE`, `WordPiece`, and `Unigram` (la lista completa [aquí](https://huggingface.co/docs/tokenizers/api/models)).
- `trainers` contiene todos los distintos tipos de `Trainer` que puedes usar para entrenar tu modelo en un corpus (uno por cada tipo de modelo; la lista completa [aquí](https://huggingface.co/docs/tokenizers/api/trainers)).
- `post_processors` contiene varios tipos de `PostProcessor` que puedes usar (la lista completa [aquí](https://huggingface.co/docs/tokenizers/api/post-processors)).
- `decoders` contiene varios tipos de `Decoder` que puedes usar para decodificar las salidas de la tokenización (la lista completa [aquí](https://huggingface.co/docs/tokenizers/components#decoders)).

Puedes encontrar la lista completas de las unidades más básicas [aquí](https://huggingface.co/docs/tokenizers/components).

## Adquirir un corpus[[acquiring-a-corpus]]

Para entrenar nuestro nuevo tokenizador, usaremos un pequeño corpus de texto (para que los ejemplos se ejecuten rápido). Los pasos para adquirir el corpus son similares a los que tomamos al [beginning of this chapter](/course/chapter6/2), pero esta vez usaremos el conjunto de datos [WikiText-2](https://huggingface.co/datasets/wikitext):

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```

La función `get_training_corpus()` es un generador que entregará lotes de 1.000 textos, los cuales usaremos para entrenar el tokenizador.

🤗 Tokenizers puedes también ser entrenada en archivos de textos directamente. Así es como podemos generar un archivo de texto conteniendo todos los textos/entradas de WikiText-2 que podemos usar localmente:

```python
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
```

A continuación mostraremos como construir tu propios propios tokenizadores BERT, GPT-2 y XLNet, bloque por bloque. Esto nos dará un ejemplo de cada una de los tres principales algoritmos de tokenización: WordPiece, BPE y Unigram. Empecemos con BERT!

## Construyendo un tokenizador WordPiece desde cero[[building-a-wordpiece-tokenizer-from-scratch]]

Para construir un tokenizador con la librería 🤗 Tokenizers, empezamos instanciando un objeto `Tokenizer` con un `model`, luego fijamos sus atributos `normalizer`, `pre_tokenizer`, `post_processor`, y `decoder` a los valores que queremos.

Para este ejemplo, crearemos un `Tokenizer` con modelo WordPiece:

```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```

Tenemos que especificar el `unk_token` para que el modelo sepa que retornar si encuentra caracteres que no ha visto antes. Otros argumentos que podemos fijar acá incluyen el `vocab` de nuestro modelo (vamos a entrenar el modelo, por lo que no necesitamos fijar esto) y `max_input_chars_per_word`, el cual especifica el largo máximo para cada palabra (palabras más largas que el valor pasado se serpararán).

El primer paso de la tokenización es la normalizacion, así que empecemos con eso. Dado que BERT es ampliamente usado, hay un `BertNormalizer` con opciones clásicas que podemos fijar para BERT: `lowercase` (transformar a minúsculas) y `strip_accents` (eliminar acentos); `clean_text` para remover todos los caracteres de control y reemplazar espacios repetidos en uno solo; y `handle_chinese_chars` el cual coloca espacios alrededor de los caracteres en Chino. Para replicar el tokenizador `bert-base-uncased`, basta con fijar este normalizador:

```python
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
```

Sin embargo, en términos generales, cuando se construye un nuevo tokenizador no tendrás acceso a tan útil normalizador ya implementado en la librería 🤗 Tokenizers -- por lo que veamos como crear el normalizador BERT a mano. La librería provee un normalizador `Lowercase` y un normalizador `StripAccents`, y puedes componer varios normalizadores usando un `Sequence` (secuencia):

```python
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```

También estamos usando un normalizador Unicode `NFD`, ya que de otra manera el normalizador `StripAccents` no reconocerá apropiadamente los caracteres acentuados y por lo tanto, no los eliminará.

Como hemos visto antes, podemos usar el método `normalize_str()` del `normalizer` para chequear los efectos que tiene en un texto dado:

```python
# print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

```python out
hello how are u?
```

> [!TIP]
> **Para ir más allá** Si pruebas las dos versiones de los normalizadores previos en un string conteniendo un caracter unicode `u"\u0085"`
> de seguro notarás que los dos normalizadores no son exactamente equivalentes.
> Para no sobre-complicar demasiado la version con `normalizers.Sequence`, no hemos incluido los reemplazos usando Expresiones Regulares (Regex) que el `BertNormalizer` requiere cuando el argumento `clean_text` se fija como `True` - lo cual es el comportamiento por defecto. Pero no te preocupes, es posible obtener la misma normalización sin usar el útil `BertNormalizer` agregando dos `normalizers.Replace` a la secuencia de normalizadores.

A continuación está la etapa de pre-tokenización. De nuevo, hay un `BertPreTokenizer` pre-hecho que podemos usar:

```python
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```

O podemos constuirlo desde cero:

```python
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
```

Nota que el pre-tokenizador `Whitespace` separa en espacios en blando y todos los caracteres que no son letras, dígitos o el guión bajo/guión al piso (_), por lo que técnicamente separa en espacios en blanco y puntuación:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]
```

Si sólo quieres separar en espacios en blanco, deberías usar el pre-tokenizador `WhitespaceSplit`:

```python
pre_tokenizer = pre_tokenizers.WhitespaceSplit()
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[("Let's", (0, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre-tokenizer.', (14, 28))]
```

Al igual que con los normalizadores, puedes un `Sequence` para componer varios pre-tokenizadores:

```python
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]
```

El siguiente paso en el pipeline de tokenización es pasar las entradas a través del modelo. Ya especificamos nuestro modelo en la inicialización, pero todavía necesitamos entrenarlo, lo cual requerirá un `WordPieceTrainer`. El aspecto principal a recordar cuando se instancia un entrenador (trainer) en 🤗 Tokenizers es que necesitas pasarle todos los tokens especiales que tiene la intención de usar -- de otra manera no los agregará al vocabulario, dado que que no están en el corpus de entrenamiento:

```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
```

Al igual que especificar `vocab_size` y `special_tokens`, podemos fijar `min_frequency` (el número de veces que un token debe aparecer para ser incluido en el vocabulario) o cambiar `continuing_subword_prefix` (si queremos usar algo diferente a `##`).

Para entrenar nuestro modelo usando el iterador que definimos antes, tenemos que ejecutar el siguiente comando:

```python
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

También podemos usar archivos de texto para entrenar nuestro tokenizador, lo cual se vería así (reinicializamos el modelo con un `WordPiece` vacío de antemano):

```python
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

En ambos casos, podemos probar el tokenizador en un texto llamando al método `encode:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)
```

```python out
['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']
```

El `encoding` (codificación) obtenido es un objeto `Encoding`, el cual contiene todas las salidas necesarias del tokenizador y sus distintos atributos: `ids`, `type_ids`, `tokens`, `offsets`, `attention_mask`, `special_tokens_mask`, y `overflowing`.

El último paso en el pipeline de tokenización es el post-procesamiento. Necesitamos agregar el token `[CLS]` al inicio y el token `[SEP]` al final (o después de cada oración, si tenemos un par de oraciones). Usaremos un `TemplateProcessor` para esto, pero primero necesitamos conocer los IDs de los tokens `[CLS]` y `[SEP]` en el vocabulario:

```python
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
# print(cls_token_id, sep_token_id)
```

```python out
(2, 3)
```

Para escribir la plantilla (template) para un `TemplateProcessor`, tenemos que especificar como tratar una sóla oración y un par de oraciones. Para ambos, escribimos los tokens especiales que queremos usar; la primera oración se representa por `$A`, mientras que la segunda oración (si se está codificando un par) se representa por `$B`. Para cada uno de estos (tokens especiales y oraciones), también especificamos el ID del tipo de token correspondiente después de un dos puntos (:).

La clásica plantilla para BERT se define como sigue:

```python
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```

Nota que necesitamos pasar los IDs de los tokens especiales, para que el tokenizador pueda convertirlos apropiadamente a sus IDs.

Una vez que se agrega esto, volviendo a nuestro ejemplo anterior nos dará:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)
```

```python out
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']
```

Y en un par de oraciones, obtenemos el resultado apropiado:

```python
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
# print(encoding.tokens)
# print(encoding.type_ids)
```

```python out
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

Ya casi finalizamos de construir este tokenizador desde cero -- el último paso es incluir un decodificador:

```python
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

Probemoslo en nuestro `encoding` previo:

```python
tokenizer.decode(encoding.ids)
```

```python out
"let's test this tokenizer... on a pair of sentences."
```

Genial! Ahora podemos guardar nuestro tokenizador en un archivo JSON así:

```python
tokenizer.save("tokenizer.json")
```

Podemos cargar ese archivo en un objeto `Tokenizer` con el método `from_file()`:

```python
new_tokenizer = Tokenizer.from_file("tokenizer.json")
```

Para usar este tokenizador en 🤗 Transformers, tenemos que envolverlo en un `PreTrainedTokenizerFast`. Podemos usar una clase generica o, si nuestro tokenizador corresponde un modelo existente, usar esa clase (en este caso, `BertTokenizerFast`). Si aplicas esta lección para construir un tokenizador nuevo de paquete, tendrás que usar la primera opción.

Para envolver el tokenizador en un `PreTrainedTokenizerFast`, podemos pasar el tokenizador que construimos como un `tokenizer_object` o pasar el archivo del tokenizador que guardarmos como `tokenizer_file`. El aspecto clave a recordar es que tenemos que manualmente fijar los tokens especiales, dado que la clase no puede inferir del objeto `tokenizer` qué token es el el token de enmascaramiento (mask token), el token `[CLS]`, etc.:

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```

Si estás usando una clase de tokenizador específico (como `BertTokenizerFast`), sólo necesitarás especificar los tokens especiales diferentes a los que están por defecto (en este caso, ninguno):

```python
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```

Luego puedes usar este tokenizador como cualquier otro tokenizador de 🤗 Transformers. Puedes guardarlo con el método `save_pretrained()`, o subirlo al Hub con el método `push_to_hub()`.

Ahora que hemos visto como construir el tokenizador WordPiece, hagamos lo mismo para un tokenizador BPE.  Iremos un poco más rápido dato que conoces todos los pasos, y sólo destacaremos las diferencias.

## Construyendo un tokenizador BPE desde cero[[building-a-bpe-tokenizer-from-scratch]]

Ahora construyamos un tokenizador GPT-2. Al igual que el tokenizador BERT, empezamos inicializando un `Tokenizer` con un modelo BPE:

```python
tokenizer = Tokenizer(models.BPE())
```

También al igual que BERT, podríamos inicializar este modelo con un vocabulario si tuviéramos uno (necesitaríamos pasar el `vocab` y `merges`, en este caso), pero dado que entrenaremos desde cero, no necesitaremos hacer eso. Tampoco necesitamos especificar `unk_token` porque GPT-2 utiliza un byte-level BPE, que no lo requiere.

GPT-2 no usa un normalizador, por lo que nos saltamos este paso y vamos directo a la pre-tokenización:

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

La opción que agregada acá `ByteLevel` es para no agregar un espacio al inicio de una oración (el cuál es el valor por defecto). Podemos echar un vistazo a la pre-tokenización de un texto de ejemplo como antes:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
```

```python out
[('Let', (0, 3)), ("'s", (3, 5)), ('Ġtest', (5, 10)), ('Ġpre', (10, 14)), ('-', (14, 15)),
 ('tokenization', (15, 27)), ('!', (27, 28))]
```

A continuación está el modelo, el cual necesita entrenamiento. Para GPT-2, el único token especial es el token de final de texto (end-of-text):

```python
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

Al igual que con el `WordPieceTrainer`, junto con `vocab_size` y `special_tokens`,  podemos especificar el `min_frequency` si queremos, o si tenemos un sufijo de fín de palabra (end-of-word suffix) (como `</w>`), podemos fijarlo con `end_of_word_suffix`.

Este tokenizador también se puede entrenar en archivos de textos:

```python
tokenizer.model = models.BPE()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

Echemos un vistazo a la tokenización de un texto de muestra:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)
```

```python out
['L', 'et', "'", 's', 'Ġtest', 'Ġthis', 'Ġto', 'ken', 'izer', '.']
```

Aplicaremos el post-procesamiento byte-level para el tokenizador GPT-2 como sigue:

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
```

La opción `trim_offsets = False` indica al post-procesador que deberíamos dejar los offsets de los tokens que comiencen con 'Ġ' sin modificar: De esta manera el inicio de los offsets apuntarán al espacio antes de la palabra, no el primer caracter de la palabra (dado que el espacio es técnicamente parte del token). Miremos el resultado con el texto que acabamos de codificar, donde `'Ġtest'` el token en el índice 4:

```python
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
sentence[start:end]
```

```python out
' test'
```

Finalmente, agregamos un decodificador byte-level:

```python
tokenizer.decoder = decoders.ByteLevel()
```

y podemos chequear si funciona de manera apropiada:

```python
tokenizer.decode(encoding.ids)
```

```python out
"Let's test this tokenizer."
```

Genial! Ahora que estamos listos, podemos guardar el tokenizador como antes, y envolverlo en un `PreTrainedTokenizerFast` o `GPT2TokenizerFast` si queremos usarlo en 🤗 Transformers:

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```

o:

```python
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
```

Como en el último ejemplo, mostraremos cómo construir un tokenizador Unigram desde cero.

## Construyendo un tokenizador Unigran desde cero[[building-a-unigram-tokenizer-from-scratch]]

Construyamos un tokenizador XLNet. Al igual que los tokenizadores previos, empezamos inicializando un `Tokenizer` con un modelo Unigram:

```python
tokenizer = Tokenizer(models.Unigram())
```

De nuevo, podríamos inicializar este modelo con un vocabulario si tuvieramos uno.

Para la normalización, XLNet utiliza unos pocos reemplazos (los cuales vienen de SentencePiece):

```python
from tokenizers import Regex

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)
```

Esto reemplaza <code>``</code> y <code>''</code> con <code>"</code> y cualquier secuencia de dos o más espacios con un espacio simple, además remueve los acentos en el texto a tokenizar.

El pre-tokenizador a usar para cualquier tokenizador SentencePiece es `Metaspace`:

```python
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
```

Podemos echar un vistazo a la pre-tokenización de un texto de ejemplo como antes:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")
```

```python out
[("▁Let's", (0, 5)), ('▁test', (5, 10)), ('▁the', (10, 14)), ('▁pre-tokenizer!', (14, 29))]
```

A continuación está el modelo, el cuál necesita entrenamiento. XLNet tiene varios tokens especiales:

```python
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

Un argumento muy importante a no olvidar para el `UnigramTrainer` es el `unk_token`. También podemos pasarle otros argumentos específicos al algoritmo Unigram, tales como el `shrinking_factor` para cada paso donde removemos tokens (su valor por defecto es 0.75) o el `max_piece_length` para especificar el largo máximo de un token dado (su valor por defecto es 16).

Este tokenizador también se puede entrenar en archivos de texto:

```python
tokenizer.model = models.Unigram()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

Ahora miremos la tokenización de un texto de muestra:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)
```

```python out
['▁Let', "'", 's', '▁test', '▁this', '▁to', 'ken', 'izer', '.']
```

Una peculiariodad de XLNet es que coloca el token `<cls>` al final de la oración, con un ID de tipo de 2 (para distinguirlo de los otros tokens). Como el resultado el resultado se rellena a la izquierda (left padding). Podemos lidiar con todos los tokens especiales y el token de ID de tipo con una plantilla, al igual que BERT, pero primero tenemos que obtener los IDs de los tokens `<cls>` y `<sep>`:

```python
cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
# print(cls_token_id, sep_token_id)
```

```python out
0 1
```

La plantilla se ve así:

```python
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)
```

Y podemos probar si funciona codificando un par de oraciones:

```python
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
# print(encoding.tokens)
# print(encoding.type_ids)
```

```python out
['▁Let', "'", 's', '▁test', '▁this', '▁to', 'ken', 'izer', '.', '.', '.', '<sep>', '▁', 'on', '▁', 'a', '▁pair', 
  '▁of', '▁sentence', 's', '!', '<sep>', '<cls>']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
```

Finalmente, agregamos el decodificador `Metaspace`:

```python
tokenizer.decoder = decoders.Metaspace()
```

y estamos listos con este tokenizador! Podemos guardar el tokenizador como antes, y envolverlo en un `PreTrainedTokenizerFast` o `XLNetTokenizerFast` si queremos usarlo en 🤗 Transformers. Una cosa a notar al usar `PreTrainedTokenizerFast` es que además de los tokens especiales, necesitamos decirle a la librería 🤗 Transformers que rellene a la izquierda (agregar left padding):

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
```

O de manera alternativa:

```python
from transformers import XLNetTokenizerFast

wrapped_tokenizer = XLNetTokenizerFast(tokenizer_object=tokenizer)
```

Ahora que has visto como varias de nuestras unidades más básicas se usan para construir tokenizadores existentes, deberías ser capaz de escribir cualquier tokenizador que quieras con la librería 🤗 Tokenizers y ser capaz de usarlo en la librería 🤗 Transformers.

---

# Tokenizadores, listo![[tokenizers-check]]


Gran trabajo terminando este capítulo!

Luego de esta profundizacion en los tokenizadores, deberías:

- Ser capaz de entrenar un nuevo tokenizador usando un existente como plantilla
- Entender como usar los offsets para mapear las posiciones de los tokens a sus trozos de texto original
- Conocer las diferencias entre BPE, WordPiece y Unigram
- Ser capaz de mezclar y combinar los bloques provistos por la librería 🤗 Tokenizers para construir tu propio tokenizador
- Ser capaz de usar el tokenizador dentro de la librería 🤗 Transformers.


---



# Quiz de Final de Capítulo[[end-of-chapter-quiz]]


Probemos lo que aprendimos en este capítulo!

### 1. Cuando debería entrenar un nuevo tokenizador?


- Cuando tu conjunto de datos es similar al usado por un modelo pre-entrenado existente, y tú quieres pre-entrenar un nuevo modelo.
- Cuando tu conjunto de datos es similar al usado por un modelo pre-entrenado existente, y quieres hacerle fine-tuning a un nuevo modelo usando este modelo pre-entrenado.
- Cuando tu dataset es diferente del que se utilizó en el modelo pre-entrenado existente, y quieres pre-entrenar un nuevo modelo.
- Cuando tu dataset es diferente del que se utilizó en el modelo pre-entrenado existente, pero quieres hacer fine-tuning a un nuevo modelo usando el modelo pre-entrenado.


### 2. Cuál es la ventaja de usar un generador de listas de textos comparado con una lista de listas de textos al usar `train_new_from_iterator()`?


- Ese es el único tipo que el método `train_new_from_iterator()` acepta.
- Evitarás cargar todo el conjunto de datos en memoria de una sóla vez.
- Esto permite que la librería 🤗 Tokenizers library use multiprocesamiento.
- El tokenizador que entrenarás generará mejores textos.


### 3. Cuáles son las ventajas de utilizar un tokenizador "rápido"?


- Puede procesar las entradas/inputs más rápido que un tokenizador lento cuando empaquetas muchas entradas/inputs en lotes.
- Los tokenizadores rápidos siempre tokenizan más rápidos que sus contrapartes lentas.
- Puede aplicar relleno (padding) y truncamiento.
- Tiene algunas características adicionales que permiten mapear tokens a la porción de texto que los creó.


### 4. Como hace el pipeline `token-classification` para manejar entidades que se extienden a varios tokens?


- Las entidades con las mismas etiquetas son fusionadas en una sólo entidad.
- Hay una etiqueta para el inicio de una entidad y una etiqueta para la continuación de una entidad.
- En una palabra dada, mientas el primer token tenga una etiquera de entidad, la palabra completa es considerada etiquetada con dicha entidad.
- Cuando un token tiene la etiqueta de una entidad dada, cualquier otro token consecutivo con la misma etiqueta será considerada parte de la misma entidad, a menos que sea etiquetada como el inicio de una nueva entidad.


### 5. Cómo hace el pipeline de `question-answering` para manejar contextos largos?


- En realidad no lo hace, ya que trunca los contextos largos al largo máximo aceptado por el modelo.
- Separa el contexto en varias partes y promedia los resultados obtenidos.
- Separa el contexto en varias partes (con traslape) y encuentra el puntaje máximo para una respuesta en cada parte.
- Separa el contexto en varias partes (sin traslape, por eficiencia) y encuentra el puntaje máximo para una respuesta en cada parte.


### 6. Qué es la normalización?


- Es cualquier limpieza que el tokenizador realiza en los extos en las etapas iniciales.
- Es una técnica de aumento de datos que involucra hacer el texto más normal removiendo palabras raras.
- Es el paso final de post-procesamiento donde el tokenizador agrega los tokens especiales.
- Es cuando los embeddings se llevan a media 0 y desviación estándar 1, restando la media y dividiendo la desviación estándar.


### 7. Qué es la pre-tokenización para un tokenizador de subpalabra?


- Es un paso antes de la tokenización, donde se aplica aumento de datos (como enmascaramiento aleatorio (random masking)).
- Es el paso antes de la tokenización, donde las operaciones de limpieza deseada son aplicados al texto.
- Es el paso antes que el modelo de tokenización sea aplicado para separar la entrada/input en palabras.
- Es el paso antes de que el tokenizador se aplique para separar el input en tokens.


### 8. Selecciona las afirmaciones que aplican para el modelo de tokenización BPE.


- BPE es un algoritmo de tokenización por subpalabras que comienza con un vocabulario pequeño y aprende reglas de fusión.
- BPE es un algoritmo de tokenización que comienza con un vocabulario grande y de manera progresiva remueve tokens de él.
- El tokenizador BPE aprende reglas de fusión fusionando el parte de tokens que es el más frecuente.
- Un tokenizador BPE aprende una regla de fusión fusionando el par de tokens que maximiza un puntaje que privilegia pares frecuentes con partes individuales menos frecuentes.
- BPE tokeniza palabras en subpalabras separándolas en caracteres y luego aplicando reglas de fusión.
- BPE tokeniza palabras en subpalabras encontrando la subpalabra más larga partiendo desde el inicio que está en el vocabulario, luego repite el proceso para el resto del texto.


### 9. Selecciona las afirmaciones que aplican para el modelo de tokenizacion WordPiece.


- WordPiece es un algoritmo de tokenización de subpalabras que comienza con un vocabulario pequeño y aprende reglas de fusión.
- WordPiece un algoritmo de tokenización de subpalabras que comienza con un vocabulario grande y de manera progresiva remueve tokens de él.
- Los tokenizadores WordPiece aprenden reglas de fusión fusionando el par de tokens más frecuentes.
- Un tokenizador WordPiece aprende una regla de fusión fusionando el par de tokens que maximiza un puntaje que privilegia pares frecuentes con partes individuales menos frecuentes.
- WordPiece tokeniza palabras en subpalabras encontrando la segmentación en tokens más probable, de acuerdo al modelo.
- WordPiece tokeniza palabras en subpalabras encontrando la subpalabra máas larga partiendo desde el inicio que está en el vocabularoi, luego repite el proceso para el resto del texto.


### 10. Selecciona las afirmaciones que aplican para el modelo de tokenización Unigram.


- Unigram es un algoritmo de tokenización que comienza con un vocabulario pequeño y aprende reglas de fusión.
- Unigram es un algoritmo de tokenización que comienza con un vocabulario grande y progresivamente remueve tokens de él.
- Unigram adapta su vocabulario minimizando una pérdia calculada sobre el corpus completo.
- Unigram adapta su vocabularo manteneiendo las subpalabras más frecuentes.
- Unigram tokeniza palabras en subpalabras encontrando la segmentación en tokens más probable, de acuerdo al modelo.
- Unigram tokeniza palabras en subpalabras separandolas en caracteres, luego aplicando caracteres, luego aplicando reglas de fusión.



---

# 7. Tareas clásicas de NLP



# Introduccion[[introduction]]


En el [Capitulo 3](/course/chapter3), viste como ajustar finamente un modelo para clasificacion de texto. En este capitulo, abordaremos las siguientes tareas comunes de lenguaje que son esenciales para trabajar tanto con modelos tradicionales de NLP como con LLMs modernos:

- Clasificacion de tokens
- Modelado de lenguaje enmascarado (como BERT)
- Resumenes
- Traduccion
- Preentrenamiento de modelado de lenguaje causal (como GPT-2)
- Respuesta a preguntas

Estas tareas fundamentales forman la base de como funcionan los Modelos de Lenguaje Grande (LLMs) y entenderlas es crucial para trabajar efectivamente con los modelos de lenguaje mas avanzados de hoy.


**PyTorch:**

Para hacer esto, necesitaras aprovechar todo lo que aprendiste sobre la API `Trainer` y la biblioteca 🤗 Accelerate en el [Capitulo 3](/course/chapter3), la biblioteca 🤗 Datasets en el [Capitulo 5](/course/chapter5), y la biblioteca 🤗 Tokenizers en el [Capitulo 6](/course/chapter6). Tambien subiremos nuestros resultados al Model Hub, como hicimos en el [Capitulo 4](/course/chapter4), asi que este es realmente el capitulo donde todo se une!

Cada seccion puede leerse de forma independiente y te mostrara como entrenar un modelo con la API `Trainer` o con tu propio bucle de entrenamiento, usando 🤗 Accelerate. Sientete libre de saltar cualquier parte y enfocarte en la que mas te interese: la API `Trainer` es genial para ajustar finamente o entrenar tu modelo sin preocuparte por lo que sucede detras de escena, mientras que el bucle de entrenamiento con `Accelerate` te permitira personalizar cualquier parte que desees mas facilmente.

**TensorFlow/Keras:**

Para hacer esto, necesitaras aprovechar todo lo que aprendiste sobre entrenar modelos con la API Keras en el [Capitulo 3](/course/chapter3), la biblioteca 🤗 Datasets en el [Capitulo 5](/course/chapter5), y la biblioteca 🤗 Tokenizers en el [Capitulo 6](/course/chapter6). Tambien subiremos nuestros resultados al Model Hub, como hicimos en el [Capitulo 4](/course/chapter4), asi que este es realmente el capitulo donde todo se une!

Cada seccion puede leerse de forma independiente.


> [!TIP]
> Si lees las secciones en secuencia, notaras que tienen bastante codigo y prosa en comun. La repeticion es intencional, para permitirte sumergirte (o volver despues) a cualquier tarea que te interese y encontrar un ejemplo completo funcionando.


---



# Clasificacion de tokens[[token-classification]]


La primera aplicacion que exploraremos es la clasificacion de tokens. Esta tarea generica abarca cualquier problema que pueda formularse como "atribuir una etiqueta a cada token en una oracion", como:

- **Reconocimiento de entidades nombradas (NER)**: Encontrar las entidades (como personas, ubicaciones u organizaciones) en una oracion. Esto puede formularse como atribuir una etiqueta a cada token teniendo una clase por entidad y una clase para "sin entidad".
- **Etiquetado de partes del discurso (POS)**: Marcar cada palabra en una oracion como correspondiente a una parte particular del discurso (como sustantivo, verbo, adjetivo, etc.).
- **Chunking**: Encontrar los tokens que pertenecen a la misma entidad. Esta tarea (que puede combinarse con POS o NER) puede formularse como atribuir una etiqueta (usualmente `B-`) a cualquier token que este al comienzo de un chunk, otra etiqueta (usualmente `I-`) a tokens que estan dentro de un chunk, y una tercera etiqueta (usualmente `O`) a tokens que no pertenecen a ningun chunk.


**Video:** [Ver en YouTube](https://youtu.be/wVHdVlPScxA)


Por supuesto, hay muchos otros tipos de problemas de clasificacion de tokens; estos son solo algunos ejemplos representativos. En esta seccion, ajustaremos finamente un modelo (BERT) en una tarea de NER, que luego podra calcular predicciones como esta:

<iframe src="https://course-demos-bert-finetuned-ner.hf.space" frameBorder="0" height="350" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

<a class="flex justify-center" href="/huggingface-course/bert-finetuned-ner">
<img class="block dark:hidden lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/model-eval-bert-finetuned-ner.png" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/model-eval-bert-finetuned-ner-dark.png" alt="One-hot encoded labels for question answering."/>
</a>

Puedes encontrar el modelo que entrenaremos y subiremos al Hub y verificar sus predicciones [aqui](https://huggingface.co/huggingface-course/bert-finetuned-ner?text=My+name+is+Sylvain+and+I+work+at+Hugging+Face+in+Brooklyn).

## Preparando los datos[[preparing-the-data]]

Primero lo primero, necesitamos un conjunto de datos adecuado para la clasificacion de tokens. En esta seccion usaremos el [conjunto de datos CoNLL-2003](https://huggingface.co/datasets/conll2003), que contiene historias de noticias de Reuters.

> [!TIP]
> Siempre que tu conjunto de datos consista en textos divididos en palabras con sus etiquetas correspondientes, podras adaptar los procedimientos de procesamiento de datos descritos aqui a tu propio conjunto de datos. Consulta el [Capitulo 5](/course/chapter5) si necesitas un repaso sobre como cargar tus propios datos personalizados en un `Dataset`.

### El conjunto de datos CoNLL-2003[[the-conll-2003-dataset]]

Para cargar el conjunto de datos CoNLL-2003, usamos el metodo `load_dataset()` de la biblioteca Datasets:

```py
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")
```

Esto descargara y almacenara en cache el conjunto de datos, como vimos en el [Capitulo 3](/course/chapter3) para el conjunto de datos GLUE MRPC. Inspeccionar este objeto nos muestra las columnas presentes y la division entre los conjuntos de entrenamiento, validacion y prueba:

```py
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3453
    })
})
```

En particular, podemos ver que el conjunto de datos contiene etiquetas para las tres tareas que mencionamos anteriormente: NER, POS y chunking. Una gran diferencia con otros conjuntos de datos es que los textos de entrada no se presentan como oraciones o documentos, sino como listas de palabras (la ultima columna se llama `tokens`, pero contiene palabras en el sentido de que estas son entradas pre-tokenizadas que aun necesitan pasar por el tokenizer para la tokenizacion en subpalabras).

Veamos el primer elemento del conjunto de entrenamiento:

```py
raw_datasets["train"][0]["tokens"]
```

```python out
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```

Como queremos realizar reconocimiento de entidades nombradas, veremos las etiquetas NER:

```py
raw_datasets["train"][0]["ner_tags"]
```

```python out
[3, 0, 7, 0, 0, 0, 7, 0, 0]
```

Esas son las etiquetas como enteros listos para el entrenamiento, pero no son necesariamente utiles cuando queremos inspeccionar los datos. Como en la clasificacion de texto, podemos acceder a la correspondencia entre esos enteros y los nombres de las etiquetas mirando el atributo `features` de nuestro conjunto de datos:

```py
ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature
```

```python out
Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)
```

Entonces esta columna contiene elementos que son secuencias de `ClassLabel`s. El tipo de los elementos de la secuencia esta en el atributo `feature` de este `ner_feature`, y podemos acceder a la lista de nombres mirando el atributo `names` de ese `feature`:

```py
label_names = ner_feature.feature.names
label_names
```

```python out
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

Ya vimos estas etiquetas cuando profundizamos en el pipeline de `token-classification` en el [Capitulo 6](/course/chapter6/3), pero para un repaso rapido:

- `O` significa que la palabra no corresponde a ninguna entidad.
- `B-PER`/`I-PER` significa que la palabra corresponde al comienzo de/esta dentro de una entidad de *persona*.
- `B-ORG`/`I-ORG` significa que la palabra corresponde al comienzo de/esta dentro de una entidad de *organizacion*.
- `B-LOC`/`I-LOC` significa que la palabra corresponde al comienzo de/esta dentro de una entidad de *ubicacion*.
- `B-MISC`/`I-MISC` significa que la palabra corresponde al comienzo de/esta dentro de una entidad *miscelanea*.

Ahora decodificar las etiquetas que vimos anteriormente nos da esto:

```python
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

```python out
'EU    rejects German call to boycott British lamb .'
'B-ORG O       B-MISC O    O  O       B-MISC  O    O'
```

Y para un ejemplo mezclando etiquetas `B-` e `I-`, aqui esta lo que el mismo codigo nos da en el elemento del conjunto de entrenamiento en el indice 4:

```python out
'Germany \'s representative to the European Union \'s veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer .'
'B-LOC   O  O              O  O   B-ORG    I-ORG O  O          O         B-PER  I-PER     O    O  O         O         O      O   O         O    O         O     O    B-LOC   O     O   O          O      O   O       O'
```

Como podemos ver, las entidades que abarcan dos palabras, como "European Union" y "Werner Zwingmann", se les atribuye una etiqueta `B-` para la primera palabra y una etiqueta `I-` para la segunda.

> [!TIP]
> Tu turno! Imprime las mismas dos oraciones con sus etiquetas POS o chunking.

### Procesando los datos[[processing-the-data]]


**Video:** [Ver en YouTube](https://youtu.be/iY2AZYdZAr0)


Como de costumbre, nuestros textos necesitan ser convertidos a IDs de tokens antes de que el modelo pueda darles sentido. Como vimos en el [Capitulo 6](/course/chapter6/), una gran diferencia en el caso de las tareas de clasificacion de tokens es que tenemos entradas pre-tokenizadas. Afortunadamente, la API del tokenizer puede manejar eso bastante facilmente; solo necesitamos advertir al `tokenizer` con una bandera especial.

Para comenzar, creemos nuestro objeto `tokenizer`. Como dijimos antes, usaremos un modelo BERT preentrenado, asi que comenzaremos descargando y almacenando en cache el tokenizer asociado:

```python
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Puedes reemplazar el `model_checkpoint` con cualquier otro modelo que prefieras del [Hub](https://huggingface.co/models), o con una carpeta local en la que hayas guardado un modelo preentrenado y un tokenizer. La unica restriccion es que el tokenizer necesita estar respaldado por la biblioteca Tokenizers, para que haya una version "rapida" disponible. Puedes ver todas las arquitecturas que vienen con una version rapida en [esta gran tabla](https://huggingface.co/transformers/#supported-frameworks), y para verificar que el objeto `tokenizer` que estas usando esta efectivamente respaldado por Tokenizers puedes mirar su atributo `is_fast`:

```py
tokenizer.is_fast
```

```python out
True
```

Para tokenizar una entrada pre-tokenizada, podemos usar nuestro `tokenizer` como de costumbre y simplemente agregar `is_split_into_words=True`:

```py
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()
```

```python out
['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
```

Como podemos ver, el tokenizer agrego los tokens especiales usados por el modelo (`[CLS]` al comienzo y `[SEP]` al final) y dejo la mayoria de las palabras sin cambios. La palabra `lamb`, sin embargo, fue tokenizada en dos subpalabras, `la` y `##mb`. Esto introduce una discrepancia entre nuestras entradas y las etiquetas: la lista de etiquetas tiene solo 9 elementos, mientras que nuestra entrada ahora tiene 12 tokens. Contabilizar los tokens especiales es facil (sabemos que estan al comienzo y al final), pero tambien necesitamos asegurarnos de alinear todas las etiquetas con las palabras correctas.

Afortunadamente, como estamos usando un tokenizer rapido tenemos acceso a los superpoderes de Tokenizers, lo que significa que podemos mapear facilmente cada token a su palabra correspondiente (como se vio en el [Capitulo 6](/course/chapter6/3)):

```py
inputs.word_ids()
```

```python out
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
```

Con un poco de trabajo, podemos entonces expandir nuestra lista de etiquetas para que coincida con los tokens. La primera regla que aplicaremos es que los tokens especiales obtienen una etiqueta de `-100`. Esto es porque por defecto `-100` es un indice que se ignora en la funcion de perdida que usaremos (entropia cruzada). Luego, cada token obtiene la misma etiqueta que el token que comenzo la palabra en la que esta, ya que son parte de la misma entidad. Para tokens dentro de una palabra pero no al comienzo, reemplazamos la `B-` con `I-` (ya que el token no comienza la entidad):

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Comienzo de una nueva palabra!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Token especial
            new_labels.append(-100)
        else:
            # Misma palabra que el token anterior
            label = labels[word_id]
            # Si la etiqueta es B-XXX la cambiamos a I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
```

Probemoslo en nuestra primera oracion:

```py
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
```

```python out
[3, 0, 7, 0, 0, 0, 7, 0, 0]
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
```

Como podemos ver, nuestra funcion agrego el `-100` para los dos tokens especiales al comienzo y al final, y un nuevo `0` para nuestra palabra que fue dividida en dos tokens.

> [!TIP]
> Tu turno! Algunos investigadores prefieren atribuir solo una etiqueta por palabra, y asignar `-100` a los otros subtokens en una palabra dada. Esto es para evitar que palabras largas que se dividen en muchos subtokens contribuyan fuertemente a la perdida. Cambia la funcion anterior para alinear etiquetas con IDs de entrada siguiendo esta regla.

Para preprocesar todo nuestro conjunto de datos, necesitamos tokenizar todas las entradas y aplicar `align_labels_with_tokens()` en todas las etiquetas. Para aprovechar la velocidad de nuestro tokenizer rapido, es mejor tokenizar muchos textos al mismo tiempo, asi que escribiremos una funcion que procesa una lista de ejemplos y usa el metodo `Dataset.map()` con la opcion `batched=True`. Lo unico que es diferente de nuestro ejemplo anterior es que la funcion `word_ids()` necesita obtener el indice del ejemplo del que queremos los IDs de palabras cuando las entradas al tokenizer son listas de textos (o en nuestro caso, lista de listas de palabras), asi que agregamos eso tambien:

```py
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```

Ten en cuenta que aun no hemos rellenado nuestras entradas; haremos eso despues, cuando creemos los lotes con un data collator.

Ahora podemos aplicar todo ese preprocesamiento de una vez en las otras divisiones de nuestro conjunto de datos:

```py
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

Hemos hecho la parte mas dificil! Ahora que los datos han sido preprocesados, el entrenamiento real se parecera mucho a lo que hicimos en el [Capitulo 3](/course/chapter3).


**PyTorch:**

## Ajustando finamente el modelo con la API `Trainer`[[fine-tuning-the-model-with-the-trainer-api]]

El codigo real usando el `Trainer` sera el mismo que antes; los unicos cambios son la forma en que los datos se agrupan en un lote y la funcion de calculo de metricas.

**TensorFlow/Keras:**

## Ajustando finamente el modelo con Keras[[fine-tuning-the-model-with-keras]]

El codigo real usando Keras sera muy similar al anterior; los unicos cambios son la forma en que los datos se agrupan en un lote y la funcion de calculo de metricas.


### Agrupacion de datos[[data-collation]]

No podemos simplemente usar un `DataCollatorWithPadding` como en el [Capitulo 3](/course/chapter3) porque eso solo rellena las entradas (IDs de entrada, mascara de atencion e IDs de tipo de token). Aqui nuestras etiquetas deben ser rellenadas exactamente de la misma manera que las entradas para que permanezcan del mismo tamano, usando `-100` como valor para que las predicciones correspondientes sean ignoradas en el calculo de la perdida.

Todo esto lo hace un [`DataCollatorForTokenClassification`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorfortokenclassification). Como el `DataCollatorWithPadding`, toma el `tokenizer` usado para preprocesar las entradas:


**PyTorch:**

```py
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

**TensorFlow/Keras:**

```py
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)
```


Para probar esto en algunas muestras, simplemente podemos llamarlo en una lista de ejemplos de nuestro conjunto de entrenamiento tokenizado:

```py
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]
```

```python out
tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
        [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
```

Comparemos esto con las etiquetas del primer y segundo elemento en nuestro conjunto de datos:

```py
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])
```

```python out
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
[-100, 1, 2, -100]
```


**PyTorch:**

Como podemos ver, el segundo conjunto de etiquetas ha sido rellenado a la longitud del primero usando `-100`s.

**TensorFlow/Keras:**

Nuestro data collator esta listo para funcionar! Ahora usemoslo para hacer un `tf.data.Dataset` con el metodo `to_tf_dataset()`. Tambien puedes usar `model.prepare_tf_dataset()` para hacer esto con un poco menos de codigo repetitivo - veras esto en algunas de las otras secciones de este capitulo.

```py
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```


 Siguiente parada: el modelo en si.


{#if fw === 'tf'}

### Definiendo el modelo[[defining-the-model]]

Como estamos trabajando en un problema de clasificacion de tokens, usaremos la clase `TFAutoModelForTokenClassification`. Lo principal a recordar al definir este modelo es pasar informacion sobre el numero de etiquetas que tenemos. La forma mas facil de hacer esto es pasar ese numero con el argumento `num_labels`, pero si queremos un buen widget de inferencia funcionando como el que vimos al comienzo de esta seccion, es mejor establecer las correspondencias de etiquetas correctas en su lugar.

Deben establecerse mediante dos diccionarios, `id2label` y `label2id`, que contienen el mapeo de ID a etiqueta y viceversa:

```py
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
```

Ahora podemos simplemente pasarlos al metodo `TFAutoModelForTokenClassification.from_pretrained()`, y seran establecidos en la configuracion del modelo, luego guardados y subidos correctamente al Hub:

```py
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Como cuando definimos nuestro `TFAutoModelForSequenceClassification` en el [Capitulo 3](/course/chapter3), crear el modelo emite una advertencia de que algunos pesos no fueron usados (los de la cabeza de preentrenamiento) y algunos otros pesos son inicializados aleatoriamente (los de la nueva cabeza de clasificacion de tokens), y que este modelo deberia ser entrenado. Haremos eso en un minuto, pero primero verifiquemos que nuestro modelo tiene el numero correcto de etiquetas:

```python
model.config.num_labels
```

```python out
9
```

> [!WARNING]
> Si tienes un modelo con el numero incorrecto de etiquetas, obtendras un error oscuro cuando llames a `model.fit()` mas tarde. Esto puede ser molesto de depurar, asi que asegurate de hacer esta verificacion para confirmar que tienes el numero esperado de etiquetas.

### Ajustando finamente el modelo[[fine-tuning-the-model]]

Ahora estamos listos para entrenar nuestro modelo! Solo tenemos que hacer un par de cosas mas antes: debemos iniciar sesion en Hugging Face y definir nuestros hiperparametros de entrenamiento. Si estas trabajando en un notebook, hay una funcion conveniente para ayudarte con esto:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Esto mostrara un widget donde puedes ingresar tus credenciales de inicio de sesion de Hugging Face.

Si no estas trabajando en un notebook, simplemente escribe la siguiente linea en tu terminal:

```bash
huggingface-cli login
```

Despues de iniciar sesion, podemos preparar todo lo que necesitamos para compilar nuestro modelo. Transformers proporciona una funcion conveniente `create_optimizer()` que te dara un optimizador `AdamW` con configuraciones apropiadas para el decaimiento de peso y el decaimiento de la tasa de aprendizaje, ambos mejoraran el rendimiento de tu modelo comparado con el optimizador `Adam` incorporado:

```python
from transformers import create_optimizer
import tensorflow as tf

# Entrenar en precision mixta float16
# Comenta esta linea si estas usando una GPU que no se beneficiara de esto
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# El numero de pasos de entrenamiento es el numero de muestras en el conjunto de datos, dividido por el tamano del lote y luego multiplicado
# por el numero total de epocas. Ten en cuenta que el tf_train_dataset aqui es un tf.data.Dataset por lotes,
# no el Dataset original de Hugging Face, asi que su len() ya es num_samples // batch_size.
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
```

Ten en cuenta tambien que no proporcionamos un argumento `loss` a `compile()`. Esto es porque los modelos pueden calcular la perdida internamente - si compilas sin una perdida y proporcionas tus etiquetas en el diccionario de entrada (como hacemos en nuestros conjuntos de datos), entonces el modelo entrenara usando esa perdida interna, que sera apropiada para la tarea y el tipo de modelo que has elegido.

A continuacion, definimos un `PushToHubCallback` para subir nuestro modelo al Hub durante el entrenamiento, y ajustamos el modelo con ese callback:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-finetuned-ner", tokenizer=tokenizer)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)
```

Puedes especificar el nombre completo del repositorio al que quieres subir con el argumento `hub_model_id` (en particular, tendras que usar este argumento para subir a una organizacion). Por ejemplo, cuando subimos el modelo a la [organizacion `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/bert-finetuned-ner"`. Por defecto, el repositorio usado estara en tu espacio de nombres y tendra el nombre del directorio de salida que estableciste, por ejemplo `"cool_huggingface_user/bert-finetuned-ner"`.

> [!TIP]
> Si el directorio de salida que estas usando ya existe, necesita ser un clon local del repositorio al que quieres subir. Si no lo es, obtendras un error cuando llames a `model.fit()` y necesitaras establecer un nuevo nombre.

Ten en cuenta que mientras el entrenamiento ocurre, cada vez que el modelo se guarda (aqui, cada epoca) se sube al Hub en segundo plano. De esta manera, podras reanudar tu entrenamiento en otra maquina si es necesario.

En esta etapa, puedes usar el widget de inferencia en el Model Hub para probar tu modelo y compartirlo con tus amigos. Has ajustado finamente exitosamente un modelo en una tarea de clasificacion de tokens -- felicidades! Pero que tan bueno es realmente nuestro modelo? Deberiamos evaluar algunas metricas para averiguarlo.

{/if}


### Metricas[[metrics]]


**PyTorch:**

Para que el `Trainer` calcule una metrica en cada epoca, necesitaremos definir una funcion `compute_metrics()` que tome los arreglos de predicciones y etiquetas, y devuelva un diccionario con los nombres y valores de las metricas.

El framework tradicional usado para evaluar predicciones de clasificacion de tokens es [*seqeval*](https://github.com/chakki-works/seqeval). Para usar esta metrica, primero necesitamos instalar la biblioteca *seqeval*:

```py
!pip install seqeval
```

Luego podemos cargarla via la funcion `evaluate.load()` como hicimos en el [Capitulo 3](/course/chapter3):

**TensorFlow/Keras:**

El framework tradicional usado para evaluar predicciones de clasificacion de tokens es [*seqeval*](https://github.com/chakki-works/seqeval). Para usar esta metrica, primero necesitamos instalar la biblioteca *seqeval*:

```py
!pip install seqeval
```

Luego podemos cargarla via la funcion `evaluate.load()` como hicimos en el [Capitulo 3](/course/chapter3):


```py
import evaluate

metric = evaluate.load("seqeval")
```

Esta metrica no se comporta como la precision estandar: en realidad tomara las listas de etiquetas como cadenas, no enteros, asi que necesitaremos decodificar completamente las predicciones y etiquetas antes de pasarlas a la metrica. Veamos como funciona. Primero, obtendremos las etiquetas para nuestro primer ejemplo de entrenamiento:

```py
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels
```

```python out
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

Luego podemos crear predicciones falsas para esas simplemente cambiando el valor en el indice 2:

```py
predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])
```

Ten en cuenta que la metrica toma una lista de predicciones (no solo una) y una lista de etiquetas. Aqui esta la salida:

```python out
{'MISC': {'precision': 1.0, 'recall': 0.5, 'f1': 0.67, 'number': 2},
 'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
 'overall_precision': 1.0,
 'overall_recall': 0.67,
 'overall_f1': 0.8,
 'overall_accuracy': 0.89}
```


**PyTorch:**

Esto devuelve mucha informacion! Obtenemos la precision, recall y puntuacion F1 para cada entidad separada, asi como en general. Para nuestro calculo de metricas solo mantendremos la puntuacion general, pero sientete libre de ajustar la funcion `compute_metrics()` para devolver todas las metricas que te gustaria reportar.

Esta funcion `compute_metrics()` primero toma el argmax de los logits para convertirlos en predicciones (como de costumbre, los logits y las probabilidades estan en el mismo orden, asi que no necesitamos aplicar el softmax). Luego tenemos que convertir tanto las etiquetas como las predicciones de enteros a cadenas. Eliminamos todos los valores donde la etiqueta es `-100`, luego pasamos los resultados al metodo `metric.compute()`:

```py
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Eliminar indice ignorado (tokens especiales) y convertir a etiquetas
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

Ahora que esto esta hecho, estamos casi listos para definir nuestro `Trainer`. Solo necesitamos un `model` para ajustar finamente!

**TensorFlow/Keras:**

Esto devuelve mucha informacion! Obtenemos la precision, recall y puntuacion F1 para cada entidad separada, asi como en general. Ahora veamos que pasa si intentamos usar las predicciones reales de nuestro modelo para calcular algunas puntuaciones reales.

TensorFlow no le gusta concatenar nuestras predicciones juntas, porque tienen longitudes de secuencia variables. Esto significa que no podemos simplemente usar `model.predict()` -- pero eso no nos va a detener. Obtendremos algunas predicciones un lote a la vez y las concatenaremos en una gran lista larga mientras avanzamos, descartando los tokens `-100` que indican enmascaramiento/relleno, luego calculamos las metricas en la lista al final:

```py
import numpy as np

all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(label_names[predicted_idx])
            all_labels.append(label_names[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])
```


```python out
{'LOC': {'precision': 0.91, 'recall': 0.92, 'f1': 0.91, 'number': 1668},
 'MISC': {'precision': 0.70, 'recall': 0.79, 'f1': 0.74, 'number': 702},
 'ORG': {'precision': 0.85, 'recall': 0.90, 'f1': 0.88, 'number': 1661},
 'PER': {'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'number': 1617},
 'overall_precision': 0.87,
 'overall_recall': 0.91,
 'overall_f1': 0.89,
 'overall_accuracy': 0.97}
```

Como le fue a tu modelo, comparado con el nuestro? Si obtuviste numeros similares, tu entrenamiento fue un exito!


**PyTorch:**

### Definiendo el modelo[[defining-the-model]]

Como estamos trabajando en un problema de clasificacion de tokens, usaremos la clase `AutoModelForTokenClassification`. Lo principal a recordar al definir este modelo es pasar informacion sobre el numero de etiquetas que tenemos. La forma mas facil de hacer esto es pasar ese numero con el argumento `num_labels`, pero si queremos un buen widget de inferencia funcionando como el que vimos al comienzo de esta seccion, es mejor establecer las correspondencias de etiquetas correctas en su lugar.

Deben establecerse mediante dos diccionarios, `id2label` y `label2id`, que contienen los mapeos de ID a etiqueta y viceversa:

```py
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
```

Ahora podemos simplemente pasarlos al metodo `AutoModelForTokenClassification.from_pretrained()`, y seran establecidos en la configuracion del modelo y luego guardados y subidos correctamente al Hub:

```py
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Como cuando definimos nuestro `AutoModelForSequenceClassification` en el [Capitulo 3](/course/chapter3), crear el modelo emite una advertencia de que algunos pesos no fueron usados (los de la cabeza de preentrenamiento) y algunos otros pesos son inicializados aleatoriamente (los de la nueva cabeza de clasificacion de tokens), y que este modelo deberia ser entrenado. Haremos eso en un minuto, pero primero verifiquemos que nuestro modelo tiene el numero correcto de etiquetas:

```python
model.config.num_labels
```

```python out
9
```

> [!WARNING]
> Si tienes un modelo con el numero incorrecto de etiquetas, obtendras un error oscuro cuando llames al metodo `Trainer.train()` mas tarde (algo como "CUDA error: device-side assert triggered"). Esta es la causa numero uno de errores reportados por usuarios para tales errores, asi que asegurate de hacer esta verificacion para confirmar que tienes el numero esperado de etiquetas.

### Ajustando finamente el modelo[[fine-tuning-the-model]]

Ahora estamos listos para entrenar nuestro modelo! Solo necesitamos hacer dos ultimas cosas antes de definir nuestro `Trainer`: iniciar sesion en Hugging Face y definir nuestros argumentos de entrenamiento. Si estas trabajando en un notebook, hay una funcion conveniente para ayudarte con esto:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Esto mostrara un widget donde puedes ingresar tus credenciales de inicio de sesion de Hugging Face.

Si no estas trabajando en un notebook, simplemente escribe la siguiente linea en tu terminal:

```bash
huggingface-cli login
```

Una vez hecho esto, podemos definir nuestros `TrainingArguments`:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
```

Ya has visto la mayoria de estos antes: establecemos algunos hiperparametros (como la tasa de aprendizaje, el numero de epocas para entrenar y el decaimiento de peso), y especificamos `push_to_hub=True` para indicar que queremos guardar el modelo y evaluarlo al final de cada epoca, y que queremos subir nuestros resultados al Model Hub. Ten en cuenta que puedes especificar el nombre del repositorio al que quieres subir con el argumento `hub_model_id` (en particular, tendras que usar este argumento para subir a una organizacion). Por ejemplo, cuando subimos el modelo a la [organizacion `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/bert-finetuned-ner"` a `TrainingArguments`. Por defecto, el repositorio usado estara en tu espacio de nombres y tendra el nombre del directorio de salida que estableciste, entonces en nuestro caso sera `"sgugger/bert-finetuned-ner"`.

> [!TIP]
> Si el directorio de salida que estas usando ya existe, necesita ser un clon local del repositorio al que quieres subir. Si no lo es, obtendras un error cuando definas tu `Trainer` y necesitaras establecer un nuevo nombre.

Finalmente, simplemente pasamos todo al `Trainer` y lanzamos el entrenamiento:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()
```

Ten en cuenta que mientras el entrenamiento ocurre, cada vez que el modelo se guarda (aqui, cada epoca) se sube al Hub en segundo plano. De esta manera, podras reanudar tu entrenamiento en otra maquina si es necesario.

Una vez que el entrenamiento esta completo, usamos el metodo `push_to_hub()` para asegurarnos de subir la version mas reciente del modelo:

```py
trainer.push_to_hub(commit_message="Training complete")
```

Este comando devuelve la URL del commit que acaba de hacer, si quieres inspeccionarlo:

```python out
'https://huggingface.co/sgugger/bert-finetuned-ner/commit/26ab21e5b1568f9afeccdaed2d8715f571d786ed'
```

El `Trainer` tambien redacta una tarjeta del modelo con todos los resultados de evaluacion y la sube. En esta etapa, puedes usar el widget de inferencia en el Model Hub para probar tu modelo y compartirlo con tus amigos. Has ajustado finamente exitosamente un modelo en una tarea de clasificacion de tokens -- felicidades!

Si quieres profundizar un poco mas en el bucle de entrenamiento, ahora te mostraremos como hacer lo mismo usando Accelerate.

## Un bucle de entrenamiento personalizado[[a-custom-training-loop]]

Ahora echemos un vistazo al bucle de entrenamiento completo, para que puedas personalizar facilmente las partes que necesites. Se parecera mucho a lo que hicimos en el [Capitulo 3](/course/chapter3/4), con algunos cambios para la evaluacion.

### Preparando todo para el entrenamiento[[preparing-everything-for-training]]

Primero necesitamos construir los `DataLoader`s de nuestros conjuntos de datos. Reutilizaremos nuestro `data_collator` como un `collate_fn` y mezclaremos el conjunto de entrenamiento, pero no el conjunto de validacion:

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
```

Luego reinstanciamos nuestro modelo, para asegurarnos de que no estamos continuando el ajuste fino de antes sino comenzando desde el modelo BERT preentrenado de nuevo:

```py
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Luego necesitaremos un optimizador. Usaremos el clasico `AdamW`, que es como `Adam`, pero con una correccion en la forma en que se aplica el decaimiento de peso:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Una vez que tenemos todos esos objetos, podemos enviarlos al metodo `accelerator.prepare()`:

```py
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> Si estas entrenando en una TPU, necesitaras mover todo el codigo comenzando desde la celda de arriba a una funcion de entrenamiento dedicada. Consulta el [Capitulo 3](/course/chapter3) para mas detalles.

Ahora que hemos enviado nuestro `train_dataloader` a `accelerator.prepare()`, podemos usar su longitud para calcular el numero de pasos de entrenamiento. Recuerda que siempre debemos hacer esto despues de preparar el dataloader, ya que ese metodo cambiara su longitud. Usamos un programa lineal clasico desde la tasa de aprendizaje hasta 0:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Por ultimo, para subir nuestro modelo al Hub, necesitaremos crear un objeto `Repository` en una carpeta de trabajo. Primero inicia sesion en Hugging Face, si aun no has iniciado sesion. Determinaremos el nombre del repositorio a partir del ID del modelo que queremos darle a nuestro modelo (sientete libre de reemplazar el `repo_name` con tu propia eleccion; solo necesita contener tu nombre de usuario, que es lo que hace la funcion `get_full_repo_name()`):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'sgugger/bert-finetuned-ner-accelerate'
```

Luego podemos clonar ese repositorio en una carpeta local. Si ya existe, esta carpeta local deberia ser un clon existente del repositorio con el que estamos trabajando:

```py
output_dir = "bert-finetuned-ner-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

Ahora podemos subir cualquier cosa que guardemos en `output_dir` llamando al metodo `repo.push_to_hub()`. Esto nos ayudara a subir los modelos intermedios al final de cada epoca.

### Bucle de entrenamiento[[training-loop]]

Ahora estamos listos para escribir el bucle de entrenamiento completo. Para simplificar su parte de evaluacion, definimos esta funcion `postprocess()` que toma predicciones y etiquetas y las convierte en listas de cadenas, como nuestro objeto `metric` espera:

```py
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Eliminar indice ignorado (tokens especiales) y convertir a etiquetas
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
```

Luego podemos escribir el bucle de entrenamiento. Despues de definir una barra de progreso para seguir como va el entrenamiento, el bucle tiene tres partes:

- El entrenamiento en si, que es la iteracion clasica sobre el `train_dataloader`, paso hacia adelante a traves del modelo, luego paso hacia atras y paso del optimizador.
- La evaluacion, en la que hay una novedad despues de obtener las salidas de nuestro modelo en un lote: dado que dos procesos pueden haber rellenado las entradas y etiquetas a diferentes formas, necesitamos usar `accelerator.pad_across_processes()` para hacer las predicciones y etiquetas de la misma forma antes de llamar al metodo `gather()`. Si no hacemos esto, la evaluacion producira un error o se colgara para siempre. Luego enviamos los resultados a `metric.add_batch()` y llamamos a `metric.compute()` una vez que el bucle de evaluacion termina.
- Guardar y subir, donde primero guardamos el modelo y el tokenizer, luego llamamos a `repo.push_to_hub()`. Ten en cuenta que usamos el argumento `blocking=False` para decirle a la biblioteca Hub que suba en un proceso asincrono. De esta manera, el entrenamiento continua normalmente y esta instruccion (larga) se ejecuta en segundo plano.

Aqui esta el codigo completo para el bucle de entrenamiento:

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Entrenamiento
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluacion
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necesario para rellenar predicciones y etiquetas para ser recopiladas
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Guardar y subir
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
```

En caso de que esta sea la primera vez que ves un modelo guardado con Accelerate, tomemos un momento para inspeccionar las tres lineas de codigo que lo acompanan:

```py
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

La primera linea es autoexplicativa: le dice a todos los procesos que esperen hasta que todos esten en esa etapa antes de continuar. Esto es para asegurarnos de que tenemos el mismo modelo en cada proceso antes de guardar. Luego tomamos el `unwrapped_model`, que es el modelo base que definimos. El metodo `accelerator.prepare()` cambia el modelo para que funcione en entrenamiento distribuido, por lo que ya no tendra el metodo `save_pretrained()`; el metodo `accelerator.unwrap_model()` deshace ese paso. Por ultimo, llamamos a `save_pretrained()` pero le decimos a ese metodo que use `accelerator.save()` en lugar de `torch.save()`.

Una vez hecho esto, deberias tener un modelo que produce resultados bastante similares al entrenado con el `Trainer`. Puedes verificar el modelo que entrenamos usando este codigo en [*huggingface-course/bert-finetuned-ner-accelerate*](https://huggingface.co/huggingface-course/bert-finetuned-ner-accelerate). Y si quieres probar cualquier ajuste al bucle de entrenamiento, puedes implementarlos directamente editando el codigo mostrado arriba!


## Usando el modelo ajustado finamente[[using-the-fine-tuned-model]]

Ya te mostramos como puedes usar el modelo que ajustamos finamente en el Model Hub con el widget de inferencia. Para usarlo localmente en un `pipeline`, solo tienes que especificar el identificador de modelo apropiado:

```py
from transformers import pipeline

# Reemplaza esto con tu propio checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[{'entity_group': 'PER', 'score': 0.9988506, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.9647625, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.9986118, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

Excelente! Nuestro modelo esta funcionando tan bien como el predeterminado para este pipeline!


---



# Ajuste fino de un modelo de lenguaje enmascarado[[fine-tuning-a-masked-language-model]]


Para muchas aplicaciones de PLN que involucran modelos Transformer, puedes simplemente tomar un modelo preentrenado del Hugging Face Hub y ajustarlo finamente directamente con tus datos para la tarea en cuestión. Siempre que el corpus utilizado para el preentrenamiento no sea muy diferente del corpus usado para el ajuste fino, el aprendizaje por transferencia generalmente producirá buenos resultados.

Sin embargo, hay algunos casos en los que querrás primero ajustar finamente el modelo de lenguaje con tus datos, antes de entrenar una cabeza específica para la tarea. Por ejemplo, si tu conjunto de datos contiene contratos legales o artículos científicos, un modelo Transformer estándar como BERT típicamente tratará las palabras específicas del dominio en tu corpus como tokens raros, y el rendimiento resultante puede ser menos que satisfactorio. Al ajustar finamente el modelo de lenguaje con datos del dominio puedes mejorar el rendimiento de muchas tareas posteriores, ¡lo que significa que generalmente solo tienes que hacer este paso una vez!

Este proceso de ajustar finamente un modelo de lenguaje preentrenado con datos del dominio se llama usualmente _adaptación de dominio_. Fue popularizado en 2018 por [ULMFiT](https://arxiv.org/abs/1801.06146), que fue una de las primeras arquitecturas neuronales (basada en LSTMs) en hacer que el aprendizaje por transferencia realmente funcionara para PLN. Un ejemplo de adaptación de dominio con ULMFiT se muestra en la imagen a continuación; en esta sección haremos algo similar, ¡pero con un Transformer en lugar de un LSTM!

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/ulmfit.svg" alt="ULMFiT."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/ulmfit-dark.svg" alt="ULMFiT."/>
</div>

Al final de esta sección tendrás un [modelo de lenguaje enmascarado](https://huggingface.co/huggingface-course/distilbert-base-uncased-finetuned-imdb?text=This+is+a+great+%5BMASK%5D.) en el Hub que puede autocompletar oraciones como se muestra a continuación:

<iframe src="https://course-demos-distilbert-base-uncased-finetuned-imdb.hf.space" frameBorder="0" height="300" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

¡Vamos a ello!


**Video:** [Ver en YouTube](https://youtu.be/mqElG5QJWUg)


> [!TIP]
> Si los términos "modelado de lenguaje enmascarado" y "modelo preentrenado" te resultan desconocidos, consulta el [Capítulo 1](/course/chapter1), donde explicamos todos estos conceptos fundamentales, ¡con videos incluidos!

## Eligiendo un modelo preentrenado para modelado de lenguaje enmascarado[[picking-a-pretrained-model-for-masked-language-modeling]]

Para comenzar, vamos a elegir un modelo preentrenado adecuado para el modelado de lenguaje enmascarado. Como se muestra en la siguiente captura de pantalla, puedes encontrar una lista de candidatos aplicando el filtro "Fill-Mask" en el [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads):

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/mlm-models.png" alt="Hub models." width="80%"/>
</div>

Aunque los modelos de la familia BERT y RoBERTa son los más descargados, usaremos un modelo llamado [DistilBERT](https://huggingface.co/distilbert-base-uncased) que puede entrenarse mucho más rápido con poca o ninguna pérdida en el rendimiento de tareas posteriores. Este modelo fue entrenado usando una técnica especial llamada [_destilación de conocimiento_](https://en.wikipedia.org/wiki/Knowledge_distillation), donde un gran "modelo maestro" como BERT se usa para guiar el entrenamiento de un "modelo estudiante" que tiene muchos menos parámetros. Una explicación de los detalles de la destilación de conocimiento nos llevaría demasiado lejos en esta sección, pero si te interesa puedes leer todo al respecto en [_Natural Language Processing with Transformers_](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) (conocido coloquialmente como el libro de texto de Transformers).


**PyTorch:**

Procedamos a descargar DistilBERT usando la clase `AutoModelForMaskedLM`:

```python
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

Podemos ver cuántos parámetros tiene este modelo llamando al método `num_parameters()`:

```python
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")
```

```python out
'>>> DistilBERT number of parameters: 67M'
'>>> BERT number of parameters: 110M'
```

**TensorFlow/Keras:**

Procedamos a descargar DistilBERT usando la clase `AutoModelForMaskedLM`:

```python
from transformers import TFAutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

Podemos ver cuántos parámetros tiene este modelo llamando al método `summary()`:

```python
model.summary()
```

```python out
Model: "tf_distil_bert_for_masked_lm"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
distilbert (TFDistilBertMain multiple                  66362880
_________________________________________________________________
vocab_transform (Dense)      multiple                  590592
_________________________________________________________________
vocab_layer_norm (LayerNorma multiple                  1536
_________________________________________________________________
vocab_projector (TFDistilBer multiple                  23866170
=================================================================
Total params: 66,985,530
Trainable params: 66,985,530
Non-trainable params: 0
_________________________________________________________________
```


Con alrededor de 67 millones de parámetros, DistilBERT es aproximadamente dos veces más pequeño que el modelo base de BERT, lo que se traduce aproximadamente en una aceleración de dos veces en el entrenamiento -- ¡genial! Veamos ahora qué tipos de tokens predice este modelo como las completaciones más probables de una pequeña muestra de texto:

```python
text = "This is a great [MASK]."
```

Como humanos, podemos imaginar muchas posibilidades para el token `[MASK]`, como "day", "ride" o "painting". Para los modelos preentrenados, las predicciones dependen del corpus con el que se entrenó el modelo, ya que aprende a captar los patrones estadísticos presentes en los datos. Al igual que BERT, DistilBERT fue preentrenado con los conjuntos de datos de [Wikipedia en inglés](https://huggingface.co/datasets/wikipedia) y [BookCorpus](https://huggingface.co/datasets/bookcorpus), por lo que esperamos que las predicciones para `[MASK]` reflejen estos dominios. Para predecir la máscara necesitamos el tokenizador de DistilBERT para producir las entradas del modelo, así que descarguémoslo también del Hub:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Con un tokenizador y un modelo, ahora podemos pasar nuestro ejemplo de texto al modelo, extraer los logits e imprimir los 5 candidatos principales:


**PyTorch:**

```python
import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Encontrar la ubicación de [MASK] y extraer sus logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Elegir los candidatos de [MASK] con los logits más altos
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
```

**TensorFlow/Keras:**

```python
import numpy as np
import tensorflow as tf

inputs = tokenizer(text, return_tensors="np")
token_logits = model(**inputs).logits
# Encontrar la ubicación de [MASK] y extraer sus logits
mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Elegir los candidatos de [MASK] con los logits más altos
# Negamos el array antes de argsort para obtener los más grandes, no los más pequeños, logits
top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")
```


```python out
'>>> This is a great deal.'
'>>> This is a great success.'
'>>> This is a great adventure.'
'>>> This is a great idea.'
'>>> This is a great feat.'
```

Podemos ver en los resultados que las predicciones del modelo se refieren a términos cotidianos, lo cual quizás no es sorprendente dada la base de Wikipedia en inglés. ¡Veamos cómo podemos cambiar este dominio a algo un poco más especializado -- reseñas de películas altamente polarizadas!


## El conjunto de datos[[the-dataset]]

Para demostrar la adaptación de dominio, usaremos el famoso [Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (o IMDb para abreviar), que es un corpus de reseñas de películas que se usa frecuentemente para evaluar modelos de análisis de sentimientos. Al ajustar finamente DistilBERT con este corpus, esperamos que el modelo de lenguaje adapte su vocabulario de los datos factuales de Wikipedia con los que fue preentrenado a los elementos más subjetivos de las reseñas de películas. Podemos obtener los datos del Hugging Face Hub con la función `load_dataset()` de Datasets:

```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
imdb_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```

Podemos ver que las divisiones `train` y `test` consisten cada una de 25,000 reseñas, mientras que hay una división sin etiquetar llamada `unsupervised` que contiene 50,000 reseñas. Echemos un vistazo a algunas muestras para tener una idea de qué tipo de texto estamos tratando. Como hemos hecho en capítulos anteriores del curso, encadenaremos las funciones `Dataset.shuffle()` y `Dataset.select()` para crear una muestra aleatoria:

```python
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
```

```python out

'>>> Review: This is your typical Priyadarshan movie--a bunch of loony characters out on some silly mission. His signature climax has the entire cast of the film coming together and fighting each other in some crazy moshpit over hidden money. Whether it is a winning lottery ticket in Malamaal Weekly, black money in Hera Pheri, "kodokoo" in Phir Hera Pheri, etc., etc., the director is becoming ridiculously predictable. Don\'t get me wrong; as clichéd and preposterous his movies may be, I usually end up enjoying the comedy. However, in most his previous movies there has actually been some good humor, (Hungama and Hera Pheri being noteworthy ones). Now, the hilarity of his films is fading as he is using the same formula over and over again.<br /><br />Songs are good. Tanushree Datta looks awesome. Rajpal Yadav is irritating, and Tusshar is not a whole lot better. Kunal Khemu is OK, and Sharman Joshi is the best.'
'>>> Label: 0'

'>>> Review: Okay, the story makes no sense, the characters lack any dimensionally, the best dialogue is ad-libs about the low quality of movie, the cinematography is dismal, and only editing saves a bit of the muddle, but Sam" Peckinpah directed the film. Somehow, his direction is not enough. For those who appreciate Peckinpah and his great work, this movie is a disappointment. Even a great cast cannot redeem the time the viewer wastes with this minimal effort.<br /><br />The proper response to the movie is the contempt that the director San Peckinpah, James Caan, Robert Duvall, Burt Young, Bo Hopkins, Arthur Hill, and even Gig Young bring to their work. Watch the great Peckinpah films. Skip this mess.'
'>>> Label: 0'

'>>> Review: I saw this movie at the theaters when I was about 6 or 7 years old. I loved it then, and have recently come to own a VHS version. <br /><br />My 4 and 6 year old children love this movie and have been asking again and again to watch it. <br /><br />I have enjoyed watching it again too. Though I have to admit it is not as good on a little TV.<br /><br />I do not have older children so I do not know what they would think of it. <br /><br />The songs are very cute. My daughter keeps singing them over and over.<br /><br />Hope this helps.'
'>>> Label: 1'
```

Sí, estas son ciertamente reseñas de películas, y si tienes la edad suficiente quizás incluso entiendas el comentario en la última reseña sobre tener una versión en VHS. Aunque no necesitaremos las etiquetas para el modelado de lenguaje, ya podemos ver que un `0` denota una reseña negativa, mientras que un `1` corresponde a una positiva.

> [!TIP]
> Pruébalo: Crea una muestra aleatoria de la división `unsupervised` y verifica que las etiquetas no sean ni `0` ni `1`. Mientras lo haces, también podrías verificar que las etiquetas en las divisiones `train` y `test` sean efectivamente `0` o `1` -- ¡esta es una verificación de cordura útil que todo practicante de PLN debería realizar al comienzo de un nuevo proyecto!

Ahora que hemos echado un vistazo rápido a los datos, profundicemos en prepararlos para el modelado de lenguaje enmascarado. Como veremos, hay algunos pasos adicionales que se necesitan tomar en comparación con las tareas de clasificación de secuencias que vimos en el [Capítulo 3](/course/chapter3). ¡Vamos!

## Preprocesamiento de los datos[[preprocessing-the-data]]


**Video:** [Ver en YouTube](https://youtu.be/8PmhEIXhBvI)


Tanto para el modelado de lenguaje auto-regresivo como para el enmascarado, un paso de preprocesamiento común es concatenar todos los ejemplos y luego dividir el corpus completo en fragmentos de igual tamaño. Esto es bastante diferente de nuestro enfoque habitual, donde simplemente tokenizamos ejemplos individuales. ¿Por qué concatenar todo junto? La razón es que los ejemplos individuales podrían truncarse si son demasiado largos, y eso resultaría en perder información que podría ser útil para la tarea de modelado de lenguaje.

Entonces, para comenzar, primero tokenizaremos nuestro corpus como de costumbre, pero _sin_ establecer la opción `truncation=True` en nuestro tokenizador. También obtendremos los IDs de palabras si están disponibles (lo cual será así si estamos usando un tokenizador rápido, como se describe en el [Capítulo 6](/course/chapter6/3)), ya que los necesitaremos más adelante para hacer el enmascaramiento de palabras completas. Envolveremos esto en una función simple, y mientras estamos en ello eliminaremos las columnas `text` y `label` ya que no las necesitamos más:

```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# ¡Usa batched=True para activar el multithreading rápido!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 50000
    })
})
```

Dado que DistilBERT es un modelo tipo BERT, podemos ver que los textos codificados consisten en los `input_ids` y `attention_mask` que hemos visto en otros capítulos, así como los `word_ids` que agregamos.

Ahora que hemos tokenizado nuestras reseñas de películas, el siguiente paso es agruparlas todas juntas y dividir el resultado en fragmentos. Pero, ¿qué tan grandes deberían ser estos fragmentos? Esto estará determinado en última instancia por la cantidad de memoria GPU que tengas disponible, pero un buen punto de partida es ver cuál es el tamaño máximo de contexto del modelo. Esto se puede inferir inspeccionando el atributo `model_max_length` del tokenizador:

```python
tokenizer.model_max_length
```

```python out
512
```

Este valor se deriva del archivo *tokenizer_config.json* asociado con un checkpoint; en este caso podemos ver que el tamaño del contexto es de 512 tokens, igual que con BERT.

> [!TIP]
> Pruébalo: Algunos modelos Transformer, como [BigBird](https://huggingface.co/google/bigbird-roberta-base) y [Longformer](hf.co/allenai/longformer-base-4096), tienen una longitud de contexto mucho mayor que BERT y otros modelos Transformer tempranos. Instancia el tokenizador para uno de estos checkpoints y verifica que el `model_max_length` coincida con lo que se indica en su tarjeta del modelo.

Entonces, para ejecutar nuestros experimentos en GPUs como las que se encuentran en Google Colab, elegiremos algo un poco más pequeño que quepa en memoria:

```python
chunk_size = 128
```

> [!WARNING]
> Ten en cuenta que usar un tamaño de fragmento pequeño puede ser perjudicial en escenarios del mundo real, así que deberías usar un tamaño que corresponda al caso de uso al que aplicarás tu modelo.

Ahora viene la parte divertida. Para mostrar cómo funciona la concatenación, tomemos algunas reseñas de nuestro conjunto de entrenamiento tokenizado e imprimamos el número de tokens por reseña:

```python
# El slicing produce una lista de listas para cada característica
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
```

```python out
'>>> Review 0 length: 200'
'>>> Review 1 length: 559'
'>>> Review 2 length: 192'
```

Luego podemos concatenar todos estos ejemplos con una simple comprensión de diccionario, de la siguiente manera:

```python
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
```

```python out
'>>> Concatenated reviews length: 951'
```

Genial, la longitud total cuadra -- ahora dividamos las reseñas concatenadas en fragmentos del tamaño dado por `chunk_size`. Para hacerlo, iteramos sobre las características en `concatenated_examples` y usamos una comprensión de lista para crear segmentos de cada característica. El resultado es un diccionario de fragmentos para cada característica:

```python
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
```

```python out
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 55'
```

Como puedes ver en este ejemplo, el último fragmento generalmente será más pequeño que el tamaño máximo de fragmento. Hay dos estrategias principales para lidiar con esto:

* Descartar el último fragmento si es más pequeño que `chunk_size`.
* Rellenar el último fragmento hasta que su longitud sea igual a `chunk_size`.

Tomaremos el primer enfoque aquí, así que envolvamos toda la lógica anterior en una sola función que podemos aplicar a nuestros conjuntos de datos tokenizados:

```python
def group_texts(examples):
    # Concatenar todos los textos
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Calcular la longitud de los textos concatenados
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Descartamos el último fragmento si es más pequeño que chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Dividir por fragmentos de max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Crear una nueva columna labels
    result["labels"] = result["input_ids"].copy()
    return result
```

Ten en cuenta que en el último paso de `group_texts()` creamos una nueva columna `labels` que es una copia de la columna `input_ids`. Como veremos en breve, esto es porque en el modelado de lenguaje enmascarado el objetivo es predecir tokens enmascarados aleatoriamente en el lote de entrada, y al crear una columna `labels` proporcionamos la verdad fundamental para que nuestro modelo de lenguaje aprenda.

Ahora apliquemos `group_texts()` a nuestros conjuntos de datos tokenizados usando nuestra confiable función `Dataset.map()`:

```python
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 61289
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 59905
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 122963
    })
})
```

Puedes ver que agrupar y luego fragmentar los textos ha producido muchos más ejemplos que los 25,000 originales para las divisiones `train` y `test`. Eso es porque ahora tenemos ejemplos que involucran _tokens contiguos_ que abarcan múltiples ejemplos del corpus original. Puedes ver esto explícitamente buscando los tokens especiales `[SEP]` y `[CLS]` en uno de los fragmentos:

```python
tokenizer.decode(lm_datasets["train"][1]["input_ids"])
```

```python out
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```

En este ejemplo puedes ver dos reseñas de películas superpuestas, una sobre una película de secundaria y otra sobre la falta de vivienda. Veamos también cómo se ven las etiquetas para el modelado de lenguaje enmascarado:

```python out
tokenizer.decode(lm_datasets["train"][1]["labels"])
```

```python out
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```

Como se esperaba de nuestra función `group_texts()` anterior, esto se ve idéntico a los `input_ids` decodificados -- pero entonces, ¿cómo puede nuestro modelo aprender algo? ¡Nos falta un paso clave: insertar tokens `[MASK]` en posiciones aleatorias en las entradas! Veamos cómo podemos hacer esto sobre la marcha durante el ajuste fino usando un colector de datos especial.

## Ajuste fino de DistilBERT con la API del `Trainer`[[fine-tuning-distilbert-with-the-trainer-api]]

Ajustar finamente un modelo de lenguaje enmascarado es casi idéntico a ajustar finamente un modelo de clasificación de secuencias, como hicimos en el [Capítulo 3](/course/chapter3). La única diferencia es que necesitamos un colector de datos especial que pueda enmascarar aleatoriamente algunos de los tokens en cada lote de textos. Afortunadamente, Transformers viene preparado con un `DataCollatorForLanguageModeling` dedicado precisamente para esta tarea. Solo tenemos que pasarle el tokenizador y un argumento `mlm_probability` que especifica qué fracción de tokens enmascarar. Elegiremos 15%, que es la cantidad usada para BERT y una elección común en la literatura:

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

Para ver cómo funciona el enmascaramiento aleatorio, alimentemos algunos ejemplos al colector de datos. Dado que espera una lista de `dict`s, donde cada `dict` representa un solo fragmento de texto contiguo, primero iteramos sobre el conjunto de datos antes de alimentar el lote al colector. Eliminamos la clave `"word_ids"` para este colector de datos ya que no la espera:

```python
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

```python output
'>>> [CLS] bromwell [MASK] is a cartoon comedy. it ran at the same [MASK] as some other [MASK] about school life, [MASK] as " teachers ". [MASK] [MASK] [MASK] in the teaching [MASK] lead [MASK] to believe that bromwell high\'[MASK] satire is much closer to reality than is " teachers ". the scramble [MASK] [MASK] financially, the [MASK]ful students whogn [MASK] right through [MASK] pathetic teachers\'pomp, the pettiness of the whole situation, distinction remind me of the schools i knew and their students. when i saw [MASK] episode in [MASK] a student repeatedly tried to burn down the school, [MASK] immediately recalled. [MASK]...'

'>>> .... at.. [MASK]... [MASK]... high. a classic line plucked inspector : i\'[MASK] here to [MASK] one of your [MASK]. student : welcome to bromwell [MASK]. i expect that many adults of my age think that [MASK]mwell [MASK] is [MASK] fetched. what a pity that it isn\'t! [SEP] [CLS] [MASK]ness ( or [MASK]lessness as george 宇in stated )公 been an issue for years but never [MASK] plan to help those on the street that were once considered human [MASK] did everything from going to school, [MASK], [MASK] vote for the matter. most people think [MASK] the homeless'
```

¡Genial, funcionó! Podemos ver que el token `[MASK]` ha sido insertado aleatoriamente en varias ubicaciones de nuestro texto. Estos serán los tokens que nuestro modelo tendrá que predecir durante el entrenamiento -- ¡y lo bonito del colector de datos es que aleatorizará la inserción de `[MASK]` con cada lote!

> [!TIP]
> Pruébalo: ¡Ejecuta el fragmento de código anterior varias veces para ver el enmascaramiento aleatorio ocurrir ante tus propios ojos! También reemplaza el método `tokenizer.decode()` con `tokenizer.convert_ids_to_tokens()` para ver que a veces un solo token de una palabra dada es enmascarado, y no los otros.


**PyTorch:**

Un efecto secundario del enmascaramiento aleatorio es que nuestras métricas de evaluación no serán determinísticas cuando usemos el `Trainer`, ya que usamos el mismo colector de datos para los conjuntos de entrenamiento y prueba. Veremos más adelante, cuando veamos el ajuste fino con Accelerate, cómo podemos usar la flexibilidad de un bucle de evaluación personalizado para congelar la aleatoriedad.


Al entrenar modelos para el modelado de lenguaje enmascarado, una técnica que se puede usar es enmascarar palabras completas juntas, no solo tokens individuales. Este enfoque se llama _enmascaramiento de palabras completas_. Si queremos usar el enmascaramiento de palabras completas, necesitaremos construir un colector de datos nosotros mismos. Un colector de datos es simplemente una función que toma una lista de muestras y las convierte en un lote, ¡así que hagamos esto ahora! Usaremos los IDs de palabras calculados anteriormente para hacer un mapeo entre índices de palabras y los tokens correspondientes, luego decidiremos aleatoriamente qué palabras enmascarar y aplicar esa máscara en las entradas. Ten en cuenta que las etiquetas son todas `-100` excepto las correspondientes a las palabras enmascaradas.


**PyTorch:**

```py
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Crear un mapeo entre palabras e índices de tokens correspondientes
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Enmascarar palabras aleatoriamente
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
```

**TensorFlow/Keras:**

```py
import collections
import numpy as np

from transformers.data.data_collator import tf_default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Crear un mapeo entre palabras e índices de tokens correspondientes
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Enmascarar palabras aleatoriamente
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return tf_default_data_collator(features)
```


A continuación, podemos probarlo con las mismas muestras de antes:

```py
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

```python out
'>>> [CLS] bromwell high is a cartoon comedy [MASK] it ran at the same time as some other programs about school life, such as " teachers ". my 35 years in the teaching profession lead me to believe that bromwell high\'s satire is much closer to reality than is " teachers ". the scramble to survive financially, the insightful students who can see right through their pathetic teachers\'pomp, the pettiness of the whole situation, all remind me of the schools i knew and their students. when i saw the episode in which a student repeatedly tried to burn down the school, i immediately recalled.....'

'>>> .... [MASK] [MASK] [MASK] [MASK]....... high. a classic line : inspector : i\'m here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn\'t! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless'
```

> [!TIP]
> Pruébalo: ¡Ejecuta el fragmento de código anterior varias veces para ver el enmascaramiento aleatorio ocurrir ante tus propios ojos! También reemplaza el método `tokenizer.decode()` con `tokenizer.convert_ids_to_tokens()` para ver que los tokens de una palabra dada siempre se enmascaran juntos.

Ahora que tenemos dos colectores de datos, el resto de los pasos de ajuste fino son estándar. El entrenamiento puede tomar un tiempo en Google Colab si no tienes la suerte de conseguir una mítica GPU P100, así que primero reduciremos el tamaño del conjunto de entrenamiento a unos pocos miles de ejemplos. ¡No te preocupes, todavía obtendremos un modelo de lenguaje bastante decente! Una forma rápida de reducir un conjunto de datos en Datasets es mediante la función `Dataset.train_test_split()` que vimos en el [Capítulo 5](/course/chapter5):

```python
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 1000
    })
})
```

Esto ha creado automáticamente nuevas divisiones `train` y `test`, con el tamaño del conjunto de entrenamiento establecido en 10,000 ejemplos y la validación en el 10% de eso -- ¡siéntete libre de aumentar esto si tienes una GPU potente! Lo siguiente que necesitamos hacer es iniciar sesión en el Hugging Face Hub. Si estás ejecutando este código en un notebook, puedes hacerlo con la siguiente función de utilidad:

```python
from huggingface_hub import notebook_login

notebook_login()
```

que mostrará un widget donde puedes ingresar tus credenciales. Alternativamente, puedes ejecutar:

```
huggingface-cli login
```

en tu terminal favorita e iniciar sesión allí.

{#if fw === 'tf'}

Una vez que hayamos iniciado sesión, podemos crear nuestros conjuntos de datos `tf.data`. Para hacerlo, usaremos el método `prepare_tf_dataset()`, que usa nuestro modelo para inferir automáticamente qué columnas deberían ir en el conjunto de datos. Si quieres controlar exactamente qué columnas usar, puedes usar el método `Dataset.to_tf_dataset()` en su lugar. Para mantener las cosas simples, usaremos el colector de datos estándar aquí, pero también puedes probar el colector de enmascaramiento de palabras completas y comparar los resultados como ejercicio:

```python
tf_train_dataset = model.prepare_tf_dataset(
    downsampled_dataset["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)

tf_eval_dataset = model.prepare_tf_dataset(
    downsampled_dataset["test"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)
```

A continuación, configuramos nuestros hiperparámetros de entrenamiento y compilamos nuestro modelo. Usamos la función `create_optimizer()` de la biblioteca Transformers, que nos da un optimizador `AdamW` con decaimiento lineal de la tasa de aprendizaje. También usamos la pérdida incorporada del modelo, que es el valor predeterminado cuando no se especifica una pérdida como argumento de `compile()`, y establecemos la precisión de entrenamiento en `"mixed_float16"`. Ten en cuenta que si estás usando una GPU de Colab u otra GPU que no tenga soporte acelerado de float16, probablemente deberías comentar esa línea.

Además, configuramos un `PushToHubCallback` que guardará el modelo en el Hub después de cada época. Puedes especificar el nombre del repositorio al que quieres hacer push con el argumento `hub_model_id` (en particular, tendrás que usar este argumento para hacer push a una organización). Por ejemplo, para hacer push del modelo a la [organización `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/distilbert-finetuned-imdb"`. Por defecto, el repositorio usado estará en tu namespace y tendrá el nombre del directorio de salida que estableciste, así que en nuestro caso será `"lewtun/distilbert-finetuned-imdb"`.

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Entrenar en precisión mixta float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model_name = model_checkpoint.split("/")[-1]
callback = PushToHubCallback(
    output_dir=f"{model_name}-finetuned-imdb", tokenizer=tokenizer
)
```

Ahora estamos listos para ejecutar `model.fit()` -- pero antes de hacerlo echemos un vistazo breve a la _perplejidad_, que es una métrica común para evaluar el rendimiento de los modelos de lenguaje.

{:else}

Una vez que hayamos iniciado sesión, podemos especificar los argumentos para el `Trainer`:

```python
from transformers import TrainingArguments

batch_size = 64
# Mostrar la pérdida de entrenamiento con cada época
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)
```

Aquí hemos ajustado algunas de las opciones predeterminadas, incluyendo `logging_steps` para asegurar que rastreamos la pérdida de entrenamiento con cada época. También hemos usado `fp16=True` para habilitar el entrenamiento de precisión mixta, lo que nos da otro impulso de velocidad. Por defecto, el `Trainer` eliminará cualquier columna que no sea parte del método `forward()` del modelo. Esto significa que si estás usando el colector de enmascaramiento de palabras completas, también necesitarás establecer `remove_unused_columns=False` para asegurar que no perdamos la columna `word_ids` durante el entrenamiento.

Ten en cuenta que puedes especificar el nombre del repositorio al que quieres hacer push con el argumento `hub_model_id` (en particular, tendrás que usar este argumento para hacer push a una organización). Por ejemplo, cuando hicimos push del modelo a la [organización `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/distilbert-finetuned-imdb"` a `TrainingArguments`. Por defecto, el repositorio usado estará en tu namespace y tendrá el nombre del directorio de salida que estableciste, así que en nuestro caso será `"lewtun/distilbert-finetuned-imdb"`.

Ahora tenemos todos los ingredientes para instanciar el `Trainer`. Aquí solo usamos el `data_collator` estándar, pero puedes probar el colector de enmascaramiento de palabras completas y comparar los resultados como ejercicio:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Ahora estamos listos para ejecutar `trainer.train()` -- pero antes de hacerlo echemos un vistazo breve a la _perplejidad_, que es una métrica común para evaluar el rendimiento de los modelos de lenguaje.

{/if}

### Perplejidad para modelos de lenguaje[[perplexity-for-language-models]]


**Video:** [Ver en YouTube](https://youtu.be/NURcDHhYe98)


A diferencia de otras tareas como clasificación de texto o respuesta a preguntas donde se nos da un corpus etiquetado para entrenar, con el modelado de lenguaje no tenemos etiquetas explícitas. Entonces, ¿cómo determinamos qué hace un buen modelo de lenguaje? Como con la función de autocorrección en tu teléfono, un buen modelo de lenguaje es uno que asigna altas probabilidades a oraciones que son gramaticalmente correctas, y bajas probabilidades a oraciones sin sentido. Para darte una mejor idea de cómo se ve esto, puedes encontrar conjuntos completos de "errores de autocorrección" en línea, donde el modelo en el teléfono de una persona ha producido algunas completaciones bastante graciosas (¡y a menudo inapropiadas!)


**PyTorch:**

Suponiendo que nuestro conjunto de prueba consiste principalmente en oraciones que son gramaticalmente correctas, entonces una forma de medir la calidad de nuestro modelo de lenguaje es calcular las probabilidades que asigna a la siguiente palabra en todas las oraciones del conjunto de prueba. Altas probabilidades indica que el modelo no está "sorprendido" o "perplejo" por los ejemplos no vistos, y sugiere que ha aprendido los patrones básicos de gramática en el lenguaje. Hay varias definiciones matemáticas de perplejidad, pero la que usaremos la define como la exponencial de la pérdida de entropía cruzada. Así, podemos calcular la perplejidad de nuestro modelo preentrenado usando la función `Trainer.evaluate()` para calcular la pérdida de entropía cruzada en el conjunto de prueba y luego tomar la exponencial del resultado:

```python
import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

**TensorFlow/Keras:**

Suponiendo que nuestro conjunto de prueba consiste principalmente en oraciones que son gramaticalmente correctas, entonces una forma de medir la calidad de nuestro modelo de lenguaje es calcular las probabilidades que asigna a la siguiente palabra en todas las oraciones del conjunto de prueba. Altas probabilidades indica que el modelo no está "sorprendido" o "perplejo" por los ejemplos no vistos, y sugiere que ha aprendido los patrones básicos de gramática en el lenguaje. Hay varias definiciones matemáticas de perplejidad, pero la que usaremos la define como la exponencial de la pérdida de entropía cruzada. Así, podemos calcular la perplejidad de nuestro modelo preentrenado usando el método `model.evaluate()` para calcular la pérdida de entropía cruzada en el conjunto de prueba y luego tomar la exponencial del resultado:

```python
import math

eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
```


```python out
>>> Perplexity: 21.75
```

Una puntuación de perplejidad más baja significa un mejor modelo de lenguaje, y podemos ver aquí que nuestro modelo inicial tiene un valor algo alto. ¡Veamos si podemos bajarlo con el ajuste fino! Para hacer eso, primero ejecutamos el bucle de entrenamiento:


**PyTorch:**

```python
trainer.train()
```

**TensorFlow/Keras:**

```python
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
```


y luego calculamos la perplejidad resultante en el conjunto de prueba como antes:


**PyTorch:**

```python
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

**TensorFlow/Keras:**

```python
eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
```


```python out
>>> Perplexity: 11.32
```

¡Genial -- esta es una reducción considerable en la perplejidad, lo que nos dice que el modelo ha aprendido algo sobre el dominio de las reseñas de películas!


**PyTorch:**

Una vez que el entrenamiento ha terminado, podemos hacer push de la tarjeta del modelo con la información de entrenamiento al Hub (los checkpoints se guardan durante el entrenamiento mismo):

```python
trainer.push_to_hub()
```


> [!TIP]
> Tu turno: Ejecuta el entrenamiento anterior después de cambiar el colector de datos al colector de enmascaramiento de palabras completas. ¿Obtienes mejores resultados?


**PyTorch:**

En nuestro caso de uso no necesitamos hacer nada especial con el bucle de entrenamiento, pero en algunos casos podrías necesitar implementar alguna lógica personalizada. Para estas aplicaciones, puedes usar Accelerate -- ¡echemos un vistazo!

## Ajuste fino de DistilBERT con Accelerate[[fine-tuning-distilbert-with-accelerate]]

Como vimos con el `Trainer`, el ajuste fino de un modelo de lenguaje enmascarado es muy similar al ejemplo de clasificación de texto del [Capítulo 3](/course/chapter3). De hecho, la única sutileza es el uso de un colector de datos especial, ¡y ya lo cubrimos anteriormente en esta sección!

Sin embargo, vimos que `DataCollatorForLanguageModeling` también aplica enmascaramiento aleatorio con cada evaluación, por lo que veremos algunas fluctuaciones en nuestras puntuaciones de perplejidad con cada ejecución de entrenamiento. Una forma de eliminar esta fuente de aleatoriedad es aplicar el enmascaramiento _una vez_ en todo el conjunto de prueba, y luego usar el colector de datos predeterminado en Transformers para recopilar los lotes durante la evaluación. Para ver cómo funciona esto, implementemos una función simple que aplica el enmascaramiento en un lote, similar a nuestro primer encuentro con `DataCollatorForLanguageModeling`:

```python
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Crear una nueva columna "masked" para cada columna en el conjunto de datos
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
```

A continuación, aplicaremos esta función a nuestro conjunto de prueba y eliminaremos las columnas no enmascaradas para poder reemplazarlas con las enmascaradas. Puedes usar el enmascaramiento de palabras completas reemplazando el `data_collator` anterior con el apropiado, en cuyo caso deberías eliminar la primera línea aquí:

```py
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```

Luego podemos configurar los dataloaders como de costumbre, pero usaremos el `default_data_collator` de Transformers para el conjunto de evaluación:

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
```

Desde aquí, seguimos los pasos estándar con Accelerate. La primera orden del día es cargar una versión fresca del modelo preentrenado:

```
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

Luego necesitamos especificar el optimizador; usaremos el estándar `AdamW`:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

Con estos objetos, ahora podemos preparar todo para el entrenamiento con el objeto `Accelerator`:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Ahora que nuestro modelo, optimizador y dataloaders están configurados, podemos especificar el programador de tasa de aprendizaje de la siguiente manera:

```python
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Solo queda una última cosa por hacer antes del entrenamiento: ¡crear un repositorio de modelo en el Hugging Face Hub! Podemos usar la biblioteca Hub para generar primero el nombre completo de nuestro repo:

```python
from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'lewtun/distilbert-base-uncased-finetuned-imdb-accelerate'
```

luego crear y clonar el repositorio usando la clase `Repository` del Hub:

```python
from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)
```

Con eso hecho, es simplemente cuestión de escribir el bucle completo de entrenamiento y evaluación:

```python
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Entrenamiento
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluación
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Guardar y subir
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
```

```python out
>>> Epoch 0: Perplexity: 11.397545307900472
>>> Epoch 1: Perplexity: 10.904909330983092
>>> Epoch 2: Perplexity: 10.729503505340409
```

¡Genial, hemos podido evaluar la perplejidad con cada época y asegurar que múltiples ejecuciones de entrenamiento sean reproducibles!


## Usando nuestro modelo ajustado finamente[[using-our-fine-tuned-model]]

Puedes interactuar con tu modelo ajustado finamente ya sea usando su widget en el Hub o localmente con el `pipeline` de Transformers. Usemos este último para descargar nuestro modelo usando el pipeline `fill-mask`:

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
```

Luego podemos alimentar al pipeline nuestro texto de ejemplo "This is a great [MASK]" y ver cuáles son las 5 predicciones principales:

```python
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
```

```python out
'>>> this is a great movie.'
'>>> this is a great film.'
'>>> this is a great story.'
'>>> this is a great movies.'
'>>> this is a great character.'
```

¡Genial -- nuestro modelo claramente ha adaptado sus pesos para predecir palabras que están más fuertemente asociadas con películas!


**Video:** [Ver en YouTube](https://youtu.be/0Oxphw4Q9fo)


Esto concluye nuestro primer experimento con el entrenamiento de un modelo de lenguaje. En la [sección 6](/course/en/chapter7/6) aprenderás cómo entrenar un modelo auto-regresivo como GPT-2 desde cero; ¡dirígete allí si quieres ver cómo puedes preentrenar tu propio modelo Transformer!

> [!TIP]
> Pruébalo: Para cuantificar los beneficios de la adaptación de dominio, ajusta finamente un clasificador en las etiquetas de IMDb tanto para el checkpoint preentrenado como para el checkpoint de DistilBERT ajustado finamente. Si necesitas un repaso sobre clasificación de texto, consulta el [Capítulo 3](/course/chapter3).


---



# Traduccion[[translation]]


Ahora vamos a sumergirnos en la traduccion. Esta es otra [tarea de secuencia a secuencia](/course/chapter1/7), lo que significa que es un problema que puede formularse como ir de una secuencia a otra. En ese sentido, el problema es bastante similar a la [resumicion](/course/chapter7/6), y podrias adaptar lo que veremos aqui a otros problemas de secuencia a secuencia como:

- **Transferencia de estilo**: Crear un modelo que *traduzca* textos escritos en un cierto estilo a otro (por ejemplo, de formal a casual o de ingles shakesperiano a ingles moderno)
- **Respuesta generativa a preguntas**: Crear un modelo que genere respuestas a preguntas, dado un contexto


**Video:** [Ver en YouTube](https://youtu.be/1JvfrvZgi6c)


Si tienes un corpus suficientemente grande de textos en dos (o mas) idiomas, puedes entrenar un nuevo modelo de traduccion desde cero como haremos en la seccion sobre [modelado de lenguaje causal](/course/chapter7/6). Sin embargo, sera mas rapido ajustar finamente un modelo de traduccion existente, ya sea uno multilingue como mT5 o mBART que quieras ajustar finamente para un par de idiomas especifico, o incluso un modelo especializado en traduccion de un idioma a otro que quieras ajustar finamente a tu corpus especifico.

En esta seccion, ajustaremos finamente un modelo Marian preentrenado para traducir del ingles al frances (ya que muchos empleados de Hugging Face hablan ambos idiomas) en el [conjunto de datos KDE4](https://huggingface.co/datasets/kde4), que es un conjunto de datos de archivos localizados para las [aplicaciones KDE](https://apps.kde.org/). El modelo que usaremos ha sido preentrenado en un gran corpus de textos en frances e ingles tomados del [conjunto de datos Opus](https://opus.nlpl.eu/), que en realidad contiene el conjunto de datos KDE4. Pero aunque el modelo preentrenado que usamos haya visto esos datos durante su preentrenamiento, veremos que podemos obtener una mejor version de el despues de ajustarlo finamente.

Una vez que hayamos terminado, tendremos un modelo capaz de hacer predicciones como esta:

<iframe src="https://course-demos-marian-finetuned-kde4-en-to-fr.hf.space" frameBorder="0" height="350" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

<a class="flex justify-center" href="/huggingface-course/marian-finetuned-kde4-en-to-fr">
<img class="block dark:hidden lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/modeleval-marian-finetuned-kde4-en-to-fr.png" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/modeleval-marian-finetuned-kde4-en-to-fr-dark.png" alt="One-hot encoded labels for question answering."/>
</a>

Como en las secciones anteriores, puedes encontrar el modelo real que entrenaremos y subiremos al Hub usando el codigo a continuacion y verificar sus predicciones [aqui](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr?text=This+plugin+allows+you+to+automatically+translate+web+pages+between+several+languages.).

## Preparando los datos[[preparing-the-data]]

Para ajustar finamente o entrenar un modelo de traduccion desde cero, necesitaremos un conjunto de datos adecuado para la tarea. Como se menciono anteriormente, usaremos el [conjunto de datos KDE4](https://huggingface.co/datasets/kde4) en esta seccion, pero puedes adaptar el codigo para usar tus propios datos con bastante facilidad, siempre y cuando tengas pares de oraciones en los dos idiomas desde y hacia los cuales quieres traducir. Consulta el [Capitulo 5](/course/chapter5) si necesitas un recordatorio de como cargar tus datos personalizados en un `Dataset`.

### El conjunto de datos KDE4[[the-kde4-dataset]]

Como de costumbre, descargamos nuestro conjunto de datos usando la funcion `load_dataset()`:

```py
from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
```

Si quieres trabajar con un par de idiomas diferente, puedes especificarlos por sus codigos. Un total de 92 idiomas estan disponibles para este conjunto de datos; puedes verlos todos expandiendo las etiquetas de idioma en su [tarjeta del conjunto de datos](https://huggingface.co/datasets/kde4).

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/language_tags.png" alt="Idiomas disponibles para el conjunto de datos KDE4." width="100%">

Echemos un vistazo al conjunto de datos:

```py
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 210173
    })
})
```

Tenemos 210,173 pares de oraciones, pero en una sola division, por lo que necesitaremos crear nuestro propio conjunto de validacion. Como vimos en el [Capitulo 5](/course/chapter5), un `Dataset` tiene un metodo `train_test_split()` que puede ayudarnos. Proporcionaremos una semilla para reproducibilidad:

```py
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 189155
    })
    test: Dataset({
        features: ['id', 'translation'],
        num_rows: 21018
    })
})
```

Podemos renombrar la clave `"test"` a `"validation"` asi:

```py
split_datasets["validation"] = split_datasets.pop("test")
```

Ahora echemos un vistazo a un elemento del conjunto de datos:

```py
split_datasets["train"][1]["translation"]
```

```python out
{'en': 'Default to expanded threads',
 'fr': 'Par defaut, developper les fils de discussion'}
```

Obtenemos un diccionario con dos oraciones en el par de idiomas que solicitamos. Una particularidad de este conjunto de datos lleno de terminos tecnicos de informatica es que todos estan completamente traducidos al frances. Sin embargo, los ingenieros franceses dejan la mayoria de las palabras especificas de informatica en ingles cuando hablan. Aqui, por ejemplo, la palabra "threads" bien podria aparecer en una oracion en frances, especialmente en una conversacion tecnica; pero en este conjunto de datos se ha traducido al mas correcto "fils de discussion". El modelo preentrenado que usamos, que ha sido preentrenado en un corpus mas grande de oraciones en frances e ingles, toma la opcion mas facil de dejar la palabra tal como esta:

```py
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
```

```python out
[{'translation_text': 'Par defaut pour les threads elargis'}]
```

Otro ejemplo de este comportamiento se puede ver con la palabra "plugin", que no es oficialmente una palabra francesa pero que la mayoria de los hablantes nativos entenderan y no se molestaran en traducir.
En el conjunto de datos KDE4, esta palabra se ha traducido al frances como el mas oficial "module d'extension":

```py
split_datasets["train"][172]["translation"]
```

```python out
{'en': 'Unable to import %1 using the OFX importer plugin. This file is not the correct format.',
 'fr': "Impossible d'importer %1 en utilisant le module d'extension d'importation OFX. Ce fichier n'a pas un format correct."}
```

Sin embargo, nuestro modelo preentrenado se queda con la palabra inglesa compacta y familiar:

```py
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
```

```python out
[{'translation_text': "Impossible d'importer %1 en utilisant le plugin d'importateur OFX. Ce fichier n'est pas le bon format."}]
```

Sera interesante ver si nuestro modelo ajustado finamente capta esas particularidades del conjunto de datos (alerta de spoiler: lo hara).


**Video:** [Ver en YouTube](https://youtu.be/0Oxphw4Q9fo)


> [!TIP]
> **Tu turno!** Otra palabra inglesa que se usa frecuentemente en frances es "email". Encuentra la primera muestra en el conjunto de datos de entrenamiento que use esta palabra. Como se traduce? Como traduce el modelo preentrenado la misma oracion en ingles?

### Procesando los datos[[processing-the-data]]


**Video:** [Ver en YouTube](https://youtu.be/XAR8jnZZuUs)


Ya deberias conocer el procedimiento: todos los textos necesitan ser convertidos en conjuntos de IDs de tokens para que el modelo pueda darles sentido. Para esta tarea, necesitaremos tokenizar tanto las entradas como los objetivos. Nuestra primera tarea es crear nuestro objeto `tokenizer`. Como se menciono antes, usaremos un modelo Marian preentrenado de ingles a frances. Si estas probando este codigo con otro par de idiomas, asegurate de adaptar el checkpoint del modelo. La organizacion [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) proporciona mas de mil modelos en multiples idiomas.

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
```

Tambien puedes reemplazar el `model_checkpoint` con cualquier otro modelo que prefieras del [Hub](https://huggingface.co/models), o una carpeta local donde hayas guardado un modelo preentrenado y un tokenizador.

> [!TIP]
> Si estas usando un tokenizador multilingue como mBART, mBART-50 o M2M100, necesitaras establecer los codigos de idioma de tus entradas y objetivos en el tokenizador configurando `tokenizer.src_lang` y `tokenizer.tgt_lang` a los valores correctos.

La preparacion de nuestros datos es bastante sencilla. Solo hay una cosa que recordar; necesitas asegurarte de que el tokenizador procese los objetivos en el idioma de salida (aqui, frances). Puedes hacer esto pasando los objetivos al argumento `text_targets` del metodo `__call__` del tokenizador.

Para ver como funciona esto, procesemos una muestra de cada idioma en el conjunto de entrenamiento:

```python
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
inputs
```

```python out
{'input_ids': [47591, 12, 9842, 19634, 9, 0], 'attention_mask': [1, 1, 1, 1, 1, 1], 'labels': [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]}
```

Como podemos ver, la salida contiene los IDs de entrada asociados con la oracion en ingles, mientras que los IDs asociados con la oracion en frances se almacenan en el campo `labels`. Si olvidas indicar que estas tokenizando etiquetas, seran tokenizadas por el tokenizador de entrada, lo cual en el caso de un modelo Marian no va a funcionar nada bien:

```python
wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))
```

```python out
['▁Par', '▁de', 'f', 'aut', ',', '▁de', 've', 'lop', 'per', '▁les', '▁fil', 's', '▁de', '▁discussion', '</s>']
['▁Par', '▁defaut', ',', '▁developper', '▁les', '▁fils', '▁de', '▁discussion', '</s>']
```

Como podemos ver, usar el tokenizador de ingles para preprocesar una oracion en frances resulta en muchos mas tokens, ya que el tokenizador no conoce ninguna palabra francesa (excepto aquellas que tambien aparecen en el idioma ingles, como "discussion").

Ya que `inputs` es un diccionario con nuestras claves habituales (IDs de entrada, mascara de atencion, etc.), el ultimo paso es definir la funcion de preprocesamiento que aplicaremos a los conjuntos de datos:

```python
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs
```

Ten en cuenta que establecimos la misma longitud maxima para nuestras entradas y salidas. Como los textos con los que estamos tratando parecen bastante cortos, usamos 128.

> [!TIP]
> Si estas usando un modelo T5 (mas especificamente, uno de los checkpoints `t5-xxx`), el modelo esperara que las entradas de texto tengan un prefijo que indique la tarea en cuestion, como `translate: English to French:`.

> [!WARNING]
> No prestamos atencion a la mascara de atencion de los objetivos, ya que el modelo no la esperara. En su lugar, las etiquetas correspondientes a un token de relleno deben establecerse en `-100` para que sean ignoradas en el calculo de la perdida. Esto lo hara nuestro colador de datos mas adelante ya que estamos aplicando relleno dinamico, pero si usas relleno aqui, debes adaptar la funcion de preprocesamiento para establecer todas las etiquetas que correspondan al token de relleno en `-100`.

Ahora podemos aplicar ese preprocesamiento de una sola vez a todas las divisiones de nuestro conjunto de datos:

```py
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
```

Ahora que los datos han sido preprocesados, estamos listos para ajustar finamente nuestro modelo preentrenado!


**PyTorch:**

## Ajuste fino del modelo con la API del Trainer[[fine-tuning-the-model-with-the-trainer-api]]

El codigo real usando el `Trainer` sera el mismo que antes, con solo un pequeno cambio: usamos un [`Seq2SeqTrainer`](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer) aqui, que es una subclase de `Trainer` que nos permitira manejar adecuadamente la evaluacion, usando el metodo `generate()` para predecir salidas a partir de las entradas. Profundizaremos en eso con mas detalle cuando hablemos del calculo de metricas.

Primero lo primero, necesitamos un modelo real para ajustar finamente. Usaremos la API habitual de `AutoModel`:

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

**TensorFlow/Keras:**

## Ajuste fino del modelo con Keras[[fine-tuning-the-model-with-keras]]

Primero lo primero, necesitamos un modelo real para ajustar finamente. Usaremos la API habitual de `AutoModel`:

```py
from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt=True)
```

<Tip warning={false}>

El checkpoint `Helsinki-NLP/opus-mt-en-fr` solo tiene pesos de PyTorch, por lo que obtendras un error si intentas cargar el modelo sin usar el argumento `from_pt=True` en el metodo `from_pretrained()`. Cuando especificas `from_pt=True`, la biblioteca descargara y convertira automaticamente los pesos de PyTorch por ti. Como puedes ver, es muy simple cambiar entre frameworks en Transformers!

</Tip>


Ten en cuenta que esta vez estamos usando un modelo que fue entrenado en una tarea de traduccion y que ya puede usarse, por lo que no hay advertencia sobre pesos faltantes o recien inicializados.

### Colacion de datos[[data-collation]]

Necesitaremos un colador de datos para manejar el relleno para el procesamiento por lotes dinamico. No podemos simplemente usar un `DataCollatorWithPadding` como en el [Capitulo 3](/course/chapter3) en este caso, porque eso solo rellena las entradas (IDs de entrada, mascara de atencion e IDs de tipo de token). Nuestras etiquetas tambien deben ser rellenadas a la longitud maxima encontrada en las etiquetas. Y, como se menciono anteriormente, el valor de relleno usado para rellenar las etiquetas debe ser `-100` y no el token de relleno del tokenizador, para asegurar que esos valores rellenados sean ignorados en el calculo de la perdida.

Todo esto lo hace un [`DataCollatorForSeq2Seq`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorforseq2seq). Como el `DataCollatorWithPadding`, toma el `tokenizer` usado para preprocesar las entradas, pero tambien toma el `model`. Esto es porque este colador de datos tambien sera responsable de preparar los IDs de entrada del decodificador, que son versiones desplazadas de las etiquetas con un token especial al principio. Ya que este desplazamiento se hace de manera ligeramente diferente para diferentes arquitecturas, el `DataCollatorForSeq2Seq` necesita conocer el objeto `model`:


**PyTorch:**

```py
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

**TensorFlow/Keras:**

```py
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
```


Para probar esto en algunas muestras, simplemente lo llamamos en una lista de ejemplos de nuestro conjunto de entrenamiento tokenizado:

```py
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
batch.keys()
```

```python out
dict_keys(['attention_mask', 'input_ids', 'labels', 'decoder_input_ids'])
```

Podemos verificar que nuestras etiquetas han sido rellenadas a la longitud maxima del lote, usando `-100`:

```py
batch["labels"]
```

```python out
tensor([[  577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,  -100,
          -100,  -100,  -100,  -100,  -100,  -100],
        [ 1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,   817,
           550,  7032,  5821,  7907, 12649,     0]])
```

Y tambien podemos echar un vistazo a los IDs de entrada del decodificador, para ver que son versiones desplazadas de las etiquetas:

```py
batch["decoder_input_ids"]
```

```python out
tensor([[59513,   577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,
         59513, 59513, 59513, 59513, 59513, 59513],
        [59513,  1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,
           817,   550,  7032,  5821,  7907, 12649]])
```

Aqui estan las etiquetas para el primer y segundo elemento en nuestro conjunto de datos:

```py
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])
```

```python out
[577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]
[1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649, 0]
```


**PyTorch:**

Pasaremos este `data_collator` al `Seq2SeqTrainer`. A continuacion, echemos un vistazo a la metrica.

**TensorFlow/Keras:**

Ahora podemos usar este `data_collator` para convertir cada uno de nuestros conjuntos de datos en un `tf.data.Dataset`, listo para el entrenamiento:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```


### Metricas[[metrics]]


**Video:** [Ver en YouTube](https://youtu.be/M05L1DhFqcw)


**PyTorch:**

La caracteristica que `Seq2SeqTrainer` agrega a su superclase `Trainer` es la capacidad de usar el metodo `generate()` durante la evaluacion o prediccion. Durante el entrenamiento, el modelo usara los `decoder_input_ids` con una mascara de atencion que asegura que no use los tokens despues del token que esta tratando de predecir, para acelerar el entrenamiento. Durante la inferencia no podremos usar esos ya que no tendremos etiquetas, por lo que es una buena idea evaluar nuestro modelo con la misma configuracion.


La metrica tradicional usada para traduccion es la [puntuacion BLEU](https://en.wikipedia.org/wiki/BLEU), introducida en [un articulo de 2002](https://aclanthology.org/P02-1040.pdf) por Kishore Papineni et al. La puntuacion BLEU evalua que tan cercanas son las traducciones a sus etiquetas. No mide la inteligibilidad o correccion gramatical de las salidas generadas por el modelo, pero usa reglas estadisticas para asegurar que todas las palabras en las salidas generadas tambien aparezcan en los objetivos. Ademas, hay reglas que penalizan las repeticiones de las mismas palabras si no se repiten tambien en los objetivos (para evitar que el modelo produzca oraciones como `"the the the the the"`) y oraciones de salida que son mas cortas que las de los objetivos (para evitar que el modelo produzca oraciones como `"the"`).

Una debilidad de BLEU es que espera que el texto ya este tokenizado, lo que dificulta comparar puntuaciones entre modelos que usan diferentes tokenizadores. Por eso, la metrica mas comunmente usada para comparar modelos de traduccion hoy en dia es [SacreBLEU](https://github.com/mjpost/sacrebleu), que aborda esta debilidad (y otras) estandarizando el paso de tokenizacion. Para usar esta metrica, primero necesitamos instalar la biblioteca SacreBLEU:

```py
!pip install sacrebleu
```

Luego podemos cargarla via `evaluate.load()` como hicimos en el [Capitulo 3](/course/chapter3):

```py
import evaluate

metric = evaluate.load("sacrebleu")
```

Esta metrica tomara textos como entradas y objetivos. Esta disenada para aceptar varios objetivos aceptables, ya que a menudo hay multiples traducciones aceptables de la misma oracion -- el conjunto de datos que estamos usando solo proporciona una, pero no es raro en PLN encontrar conjuntos de datos que dan varias oraciones como etiquetas. Entonces, las predicciones deben ser una lista de oraciones, pero las referencias deben ser una lista de listas de oraciones.

Probemos un ejemplo:

```py
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{'score': 46.750469682990165,
 'counts': [11, 6, 4, 3],
 'totals': [12, 11, 10, 9],
 'precisions': [91.67, 54.54, 40.0, 33.33],
 'bp': 0.9200444146293233,
 'sys_len': 12,
 'ref_len': 13}
```

Esto obtiene una puntuacion BLEU de 46.75, lo cual es bastante bueno -- como referencia, el modelo Transformer original en el [articulo "Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) logro una puntuacion BLEU de 41.8 en una tarea de traduccion similar entre ingles y frances! (Para mas informacion sobre las metricas individuales, como `counts` y `bp`, consulta el [repositorio de SacreBLEU](https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/metrics/bleu.py#L74).) Por otro lado, si probamos con los dos tipos malos de predicciones (muchas repeticiones o demasiado cortas) que a menudo salen de los modelos de traduccion, obtendremos puntuaciones BLEU bastante malas:

```py
predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{'score': 1.683602693167689,
 'counts': [1, 0, 0, 0],
 'totals': [4, 3, 2, 1],
 'precisions': [25.0, 16.67, 12.5, 12.5],
 'bp': 0.10539922456186433,
 'sys_len': 4,
 'ref_len': 13}
```

```py
predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{'score': 0.0,
 'counts': [2, 1, 0, 0],
 'totals': [2, 1, 0, 0],
 'precisions': [100.0, 100.0, 0.0, 0.0],
 'bp': 0.004086771438464067,
 'sys_len': 2,
 'ref_len': 13}
```

La puntuacion puede ir de 0 a 100, y mayor es mejor.

{#if fw === 'tf'}

Para ir de las salidas del modelo a textos que la metrica pueda usar, usaremos el metodo `tokenizer.batch_decode()`. Solo tenemos que limpiar todos los `-100` en las etiquetas; el tokenizador hara automaticamente lo mismo para el token de relleno. Definamos una funcion que tome nuestro modelo y un conjunto de datos y calcule metricas sobre el. Tambien vamos a usar un truco que aumenta dramaticamente el rendimiento - compilar nuestro codigo de generacion con [XLA](https://www.tensorflow.org/xla), el compilador de algebra lineal acelerada de TensorFlow. XLA aplica varias optimizaciones al grafo de computacion del modelo, y resulta en mejoras significativas de velocidad y uso de memoria. Como se describe en el [blog](https://huggingface.co/blog/tf-xla-generate) de Hugging Face, XLA funciona mejor cuando las formas de nuestras entradas no varian mucho. Para manejar esto, rellenaremos nuestras entradas a multiplos de 128, y crearemos un nuevo conjunto de datos con el colador de relleno, y luego aplicaremos el decorador `@tf.function(jit_compile=True)` a nuestra funcion de generacion, que marca toda la funcion para compilacion con XLA.

```py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128
)

tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
)


@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=128,
    )


def compute_metrics():
    all_preds = []
    all_labels = []

    for batch, labels in tqdm(tf_generate_dataset):
        predictions = generate_with_xla(batch)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = labels.numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}
```

{:else}

Para ir de las salidas del modelo a textos que la metrica pueda usar, usaremos el metodo `tokenizer.batch_decode()`. Solo tenemos que limpiar todos los `-100` en las etiquetas (el tokenizador hara automaticamente lo mismo para el token de relleno):

```py
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # En caso de que el modelo devuelva mas que los logits de prediccion
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Reemplazar -100s en las etiquetas ya que no podemos decodificarlos
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Algun post-procesamiento simple
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

{/if}

Ahora que esto esta hecho, estamos listos para ajustar finamente nuestro modelo!


### Ajuste fino del modelo[[fine-tuning-the-model]]

El primer paso es iniciar sesion en Hugging Face, para que puedas subir tus resultados al Model Hub. Hay una funcion conveniente para ayudarte con esto en un notebook:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Esto mostrara un widget donde puedes ingresar tus credenciales de inicio de sesion de Hugging Face.

Si no estas trabajando en un notebook, simplemente escribe la siguiente linea en tu terminal:

```bash
huggingface-cli login
```

{#if fw === 'tf'}

Antes de comenzar, veamos que tipo de resultados obtenemos de nuestro modelo sin ningun entrenamiento:

```py
print(compute_metrics())
```

```
{'bleu': 33.26983701454733}
```

Una vez hecho esto, podemos preparar todo lo que necesitamos para compilar y entrenar nuestro modelo. Ten en cuenta el uso de `tf.keras.mixed_precision.set_global_policy("mixed_float16")` -- esto le dira a Keras que entrene usando float16, lo que puede dar una aceleracion significativa en GPUs que lo soporten (Nvidia 20xx/V100 o mas recientes).

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# El numero de pasos de entrenamiento es el numero de muestras en el conjunto de datos, dividido por el tamano del lote y luego multiplicado
# por el numero total de epocas. Ten en cuenta que el tf_train_dataset aqui es un tf.data.Dataset por lotes,
# no el Dataset original de Hugging Face, por lo que su len() ya es num_samples // batch_size.
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Entrenar en precision mixta float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

A continuacion, definimos un `PushToHubCallback` para subir nuestro modelo al Hub durante el entrenamiento, como vimos en la [seccion 2]((/course/chapter7/2)), y luego simplemente ajustamos el modelo con ese callback:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(
    output_dir="marian-finetuned-kde4-en-to-fr", tokenizer=tokenizer
)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)
```

Ten en cuenta que puedes especificar el nombre del repositorio al que quieres subir con el argumento `hub_model_id` (en particular, tendras que usar este argumento para subir a una organizacion). Por ejemplo, cuando subimos el modelo a la [organizacion `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/marian-finetuned-kde4-en-to-fr"` a `Seq2SeqTrainingArguments`. Por defecto, el repositorio usado estara en tu espacio de nombres y se llamara segun el directorio de salida que establezcas, por lo que aqui sera `"sgugger/marian-finetuned-kde4-en-to-fr"` (que es el modelo al que enlazamos al principio de esta seccion).

> [!TIP]
> Si el directorio de salida que estas usando ya existe, necesita ser un clon local del repositorio al que quieres subir. Si no lo es, obtendras un error al llamar a `model.fit()` y necesitaras establecer un nuevo nombre.

Finalmente, veamos como se ven nuestras metricas ahora que el entrenamiento ha terminado:

```py
print(compute_metrics())
```

```
{'bleu': 57.334066271545865}
```

En esta etapa, puedes usar el widget de inferencia en el Model Hub para probar tu modelo y compartirlo con tus amigos. Has ajustado finamente con exito un modelo en una tarea de traduccion -- felicitaciones!

{:else}

Una vez hecho esto, podemos definir nuestros `Seq2SeqTrainingArguments`. Como para el `Trainer`, usamos una subclase de `TrainingArguments` que contiene algunos campos mas:

```python
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
```

Aparte de los hiperparametros habituales (como tasa de aprendizaje, numero de epocas, tamano del lote y algo de decaimiento de peso), aqui hay algunos cambios comparados con lo que vimos en las secciones anteriores:

- No establecemos ninguna evaluacion regular, ya que la evaluacion toma un tiempo; solo evaluaremos nuestro modelo una vez antes del entrenamiento y despues.
- Establecemos `fp16=True`, lo que acelera el entrenamiento en GPUs modernas.
- Establecemos `predict_with_generate=True`, como se discutio arriba.
- Usamos `push_to_hub=True` para subir el modelo al Hub al final de cada epoca.

Ten en cuenta que puedes especificar el nombre completo del repositorio al que quieres subir con el argumento `hub_model_id` (en particular, tendras que usar este argumento para subir a una organizacion). Por ejemplo, cuando subimos el modelo a la [organizacion `huggingface-course`](https://huggingface.co/huggingface-course), agregamos `hub_model_id="huggingface-course/marian-finetuned-kde4-en-to-fr"` a `Seq2SeqTrainingArguments`. Por defecto, el repositorio usado estara en tu espacio de nombres y se llamara segun el directorio de salida que establezcas, por lo que en nuestro caso sera `"sgugger/marian-finetuned-kde4-en-to-fr"` (que es el modelo al que enlazamos al principio de esta seccion).

> [!TIP]
> Si el directorio de salida que estas usando ya existe, necesita ser un clon local del repositorio al que quieres subir. Si no lo es, obtendras un error al definir tu `Seq2SeqTrainer` y necesitaras establecer un nuevo nombre.


Finalmente, simplemente pasamos todo al `Seq2SeqTrainer`:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

Antes de entrenar, primero veremos la puntuacion que obtiene nuestro modelo, para verificar que no estemos empeorando las cosas con nuestro ajuste fino. Este comando tomara un poco de tiempo, asi que puedes tomar un cafe mientras se ejecuta:

```python
trainer.evaluate(max_length=max_length)
```

```python out
{'eval_loss': 1.6964408159255981,
 'eval_bleu': 39.26865061007616,
 'eval_runtime': 965.8884,
 'eval_samples_per_second': 21.76,
 'eval_steps_per_second': 0.341}
```

Una puntuacion BLEU de 39 no esta mal, lo que refleja el hecho de que nuestro modelo ya es bueno traduciendo oraciones del ingles al frances.

A continuacion viene el entrenamiento, que tambien tomara un poco de tiempo:

```python
trainer.train()
```

Ten en cuenta que mientras ocurre el entrenamiento, cada vez que se guarda el modelo (aqui, cada epoca) se sube al Hub en segundo plano. De esta manera, podras reanudar tu entrenamiento en otra maquina si es necesario.

Una vez que el entrenamiento haya terminado, evaluamos nuestro modelo de nuevo -- esperamos ver alguna mejora en la puntuacion BLEU!

```py
trainer.evaluate(max_length=max_length)
```

```python out
{'eval_loss': 0.8558505773544312,
 'eval_bleu': 52.94161337775576,
 'eval_runtime': 714.2576,
 'eval_samples_per_second': 29.426,
 'eval_steps_per_second': 0.461,
 'epoch': 3.0}
```

Eso es una mejora de casi 14 puntos, lo cual es genial.

Finalmente, usamos el metodo `push_to_hub()` para asegurarnos de subir la ultima version del modelo. El `Trainer` tambien redacta una tarjeta del modelo con todos los resultados de evaluacion y la sube. Esta tarjeta del modelo contiene metadatos que ayudan al Model Hub a elegir el widget para la demo de inferencia. Usualmente, no hay necesidad de decir nada ya que puede inferir el widget correcto de la clase del modelo, pero en este caso, la misma clase de modelo puede usarse para todo tipo de problemas de secuencia a secuencia, por lo que especificamos que es un modelo de traduccion:

```py
trainer.push_to_hub(tags="translation", commit_message="Training complete")
```

Este comando devuelve la URL del commit que acaba de hacer, si quieres inspeccionarlo:

```python out
'https://huggingface.co/sgugger/marian-finetuned-kde4-en-to-fr/commit/3601d621e3baae2bc63d3311452535f8f58f6ef3'
```

En esta etapa, puedes usar el widget de inferencia en el Model Hub para probar tu modelo y compartirlo con tus amigos. Has ajustado finamente con exito un modelo en una tarea de traduccion -- felicitaciones!

Si quieres profundizar un poco mas en el bucle de entrenamiento, ahora te mostraremos como hacer lo mismo usando Accelerate.

{/if}


**PyTorch:**

## Un bucle de entrenamiento personalizado[[a-custom-training-loop]]

Ahora echemos un vistazo al bucle de entrenamiento completo, para que puedas personalizar facilmente las partes que necesites. Se parecera mucho a lo que hicimos en la [seccion 2](/course/chapter7/2) y el [Capitulo 3](/course/chapter3/4).

### Preparando todo para el entrenamiento[[preparing-everything-for-training]]

Ya has visto todo esto varias veces, asi que repasaremos el codigo bastante rapido. Primero construiremos los `DataLoader`s de nuestros conjuntos de datos, despues de configurar los conjuntos de datos al formato `"torch"` para obtener tensores de PyTorch:

```py
from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
```

A continuacion reinstanciamos nuestro modelo, para asegurarnos de que no estamos continuando el ajuste fino de antes sino comenzando desde el modelo preentrenado de nuevo:

```py
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

Luego necesitaremos un optimizador:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Una vez que tenemos todos esos objetos, podemos enviarlos al metodo `accelerator.prepare()`. Recuerda que si quieres entrenar en TPUs en un notebook de Colab, necesitaras mover todo este codigo a una funcion de entrenamiento, y esa no deberia ejecutar ninguna celda que instancie un `Accelerator`.

```py
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Ahora que hemos enviado nuestro `train_dataloader` a `accelerator.prepare()`, podemos usar su longitud para calcular el numero de pasos de entrenamiento. Recuerda que siempre debemos hacer esto despues de preparar el dataloader, ya que ese metodo cambiara la longitud del `DataLoader`. Usamos un programa lineal clasico desde la tasa de aprendizaje hasta 0:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Por ultimo, para subir nuestro modelo al Hub, necesitaremos crear un objeto `Repository` en una carpeta de trabajo. Primero inicia sesion en el Hugging Face Hub, si aun no has iniciado sesion. Determinaremos el nombre del repositorio a partir del ID del modelo que queremos darle a nuestro modelo (sientete libre de reemplazar el `repo_name` con tu propia eleccion; solo necesita contener tu nombre de usuario, que es lo que hace la funcion `get_full_repo_name()`):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'sgugger/marian-finetuned-kde4-en-to-fr-accelerate'
```

Luego podemos clonar ese repositorio en una carpeta local. Si ya existe, esta carpeta local debe ser un clon del repositorio con el que estamos trabajando:

```py
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

Ahora podemos subir cualquier cosa que guardemos en `output_dir` llamando al metodo `repo.push_to_hub()`. Esto nos ayudara a subir los modelos intermedios al final de cada epoca.

### Bucle de entrenamiento[[training-loop]]

Ahora estamos listos para escribir el bucle de entrenamiento completo. Para simplificar su parte de evaluacion, definimos esta funcion `postprocess()` que toma predicciones y etiquetas y las convierte en las listas de cadenas que nuestro objeto `metric` esperara:

```py
def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Reemplazar -100 en las etiquetas ya que no podemos decodificarlos.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Algun post-procesamiento simple
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels
```

El bucle de entrenamiento se parece mucho a los de la [seccion 2](/course/chapter7/2) y el [Capitulo 3](/course/chapter3), con algunas diferencias en la parte de evaluacion -- asi que enfoquemonos en eso!

Lo primero a notar es que usamos el metodo `generate()` para calcular predicciones, pero este es un metodo en nuestro modelo base, no el modelo envuelto que Accelerate creo en el metodo `prepare()`. Es por eso que primero desenvolvemos el modelo, luego llamamos a este metodo.

Lo segundo es que, como con la [clasificacion de tokens](/course/chapter7/2), dos procesos pueden haber rellenado las entradas y etiquetas a diferentes formas, por lo que usamos `accelerator.pad_across_processes()` para hacer que las predicciones y etiquetas tengan la misma forma antes de llamar al metodo `gather()`. Si no hacemos esto, la evaluacion fallara con un error o se colgara para siempre.

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Entrenamiento
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluacion
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necesario rellenar predicciones y etiquetas para ser recopiladas
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoca {epoch}, puntuacion BLEU: {results['score']:.2f}")

    # Guardar y subir
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Entrenamiento en progreso epoca {epoch}", blocking=False
        )
```

```python out
epoch 0, BLEU score: 53.47
epoch 1, BLEU score: 54.24
epoch 2, BLEU score: 54.44
```

Una vez que esto termine, deberias tener un modelo que tenga resultados bastante similares al entrenado con el `Seq2SeqTrainer`. Puedes revisar el que entrenamos usando este codigo en [*huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate*](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate). Y si quieres probar cualquier ajuste al bucle de entrenamiento, puedes implementarlos directamente editando el codigo mostrado arriba!


## Usando el modelo ajustado finamente[[using-the-fine-tuned-model]]

Ya te hemos mostrado como puedes usar el modelo que ajustamos finamente en el Model Hub con el widget de inferencia. Para usarlo localmente en un `pipeline`, solo tenemos que especificar el identificador de modelo correcto:

```py
from transformers import pipeline

# Reemplaza esto con tu propio checkpoint
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
```

```python out
[{'translation_text': 'Par defaut, developper les fils de discussion'}]
```

Como era de esperar, nuestro modelo preentrenado adapto su conocimiento al corpus en el que lo ajustamos finamente, y en lugar de dejar la palabra inglesa "threads" sola, ahora la traduce a la version oficial francesa. Lo mismo para "plugin":

```py
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
```

```python out
[{'translation_text': "Impossible d'importer %1 en utilisant le module externe d'importation OFX. Ce fichier n'est pas le bon format."}]
```

Otro gran ejemplo de adaptacion de dominio!

> [!TIP]
> **Tu turno!** Que devuelve el modelo en la muestra con la palabra "email" que identificaste antes?


---



# Resúmenes[[summarization]]


En esta sección veremos cómo los modelos Transformer pueden usarse para condensar documentos largos en resúmenes, una tarea conocida como _resumen de texto_. Esta es una de las tareas de PLN más desafiantes ya que requiere una variedad de habilidades, como comprender pasajes largos y generar texto coherente que capture los temas principales de un documento. Sin embargo, cuando se hace bien, el resumen de texto es una herramienta poderosa que puede acelerar varios procesos de negocio al aliviar la carga de los expertos en el dominio de leer documentos extensos en detalle.


**Video:** [Ver en YouTube](https://youtu.be/yHnr5Dk2zCI)


Aunque ya existen varios modelos ajustados finamente para resúmenes en el [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=summarization&sort=downloads), casi todos estos solo son adecuados para documentos en inglés. Por lo tanto, para agregar un giro en esta sección, entrenaremos un modelo bilingüe para inglés y español. Al final de esta sección, tendrás un [modelo](https://huggingface.co/huggingface-course/mt5-small-finetuned-amazon-en-es) que puede resumir reseñas de clientes como la que se muestra aquí:

<iframe src="https://course-demos-mt5-small-finetuned-amazon-en-es.hf.space" frameBorder="0" height="400" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

Como veremos, estos resúmenes son concisos porque se aprenden de los títulos que los clientes proporcionan en sus reseñas de productos. Comencemos armando un corpus bilingüe adecuado para esta tarea.

## Preparando un corpus multilingüe[[preparing-a-multilingual-corpus]]

Usaremos el [Multilingual Amazon Reviews Corpus](https://huggingface.co/datasets/amazon_reviews_multi) para crear nuestro resumidor bilingüe. Este corpus consiste en reseñas de productos de Amazon en seis idiomas y se usa típicamente para evaluar clasificadores multilingües. Sin embargo, dado que cada reseña viene acompañada de un título corto, ¡podemos usar los títulos como los resúmenes objetivo para que nuestro modelo aprenda! Para comenzar, descarguemos los subconjuntos en inglés y español desde el Hugging Face Hub:

```python
from datasets import load_dataset

spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
english_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 200000
    })
    validation: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
})
```

Como puedes ver, para cada idioma hay 200,000 reseñas para la división `train`, y 5,000 reseñas para cada una de las divisiones `validation` y `test`. La información de las reseñas que nos interesa está contenida en las columnas `review_body` y `review_title`. Veamos algunos ejemplos creando una función simple que toma una muestra aleatoria del conjunto de entrenamiento con las técnicas que aprendimos en el [Capítulo 5](/course/chapter5):

```python
def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")


show_samples(english_dataset)
```

```python out
'>> Title: Worked in front position, not rear'
'>> Review: 3 stars because these are not rear brakes as stated in the item description. At least the mount adapter only worked on the front fork of the bike that I got it for.'

'>> Title: meh'
'>> Review: Does it's job and it's gorgeous but mine is falling apart, I had to basically put it together again with hot glue'

'>> Title: Can\'t beat these for the money'
'>> Review: Bought this for handling miscellaneous aircraft parts and hanger "stuff" that I needed to organize; it really fit the bill. The unit arrived quickly, was well packaged and arrived intact (always a good sign). There are five wall mounts-- three on the top and two on the bottom. I wanted to mount it on the wall, so all I had to do was to remove the top two layers of plastic drawers, as well as the bottom corner drawers, place it when I wanted and mark it; I then used some of the new plastic screw in wall anchors (the 50 pound variety) and it easily mounted to the wall. Some have remarked that they wanted dividers for the drawers, and that they made those. Good idea. My application was that I needed something that I can see the contents at about eye level, so I wanted the fuller-sized drawers. I also like that these are the new plastic that doesn\'t get brittle and split like my older plastic drawers did. I like the all-plastic construction. It\'s heavy duty enough to hold metal parts, but being made of plastic it\'s not as heavy as a metal frame, so you can easily mount it to the wall and still load it up with heavy stuff, or light stuff. No problem there. For the money, you can\'t beat it. Best one of these I\'ve bought to date-- and I\'ve been using some version of these for over forty years.'
```

> [!TIP]
> **¡Pruébalo!** Cambia la semilla aleatoria en el comando `Dataset.shuffle()` para explorar otras reseñas en el corpus. Si hablas español, echa un vistazo a algunas de las reseñas en `spanish_dataset` para ver si los títulos también parecen resúmenes razonables.

Esta muestra demuestra la diversidad de reseñas que uno típicamente encuentra en línea, desde positivas hasta negativas (¡y todo lo que hay en medio!). Aunque el ejemplo con el título "meh" no es muy informativo, los otros títulos parecen resúmenes decentes de las reseñas mismas. Entrenar un modelo de resúmenes con las 400,000 reseñas tomaría demasiado tiempo en una sola GPU, así que en su lugar nos enfocaremos en generar resúmenes para un solo dominio de productos. Para tener una idea de qué dominios podemos elegir, convirtamos `english_dataset` a un `pandas.DataFrame` y calculemos el número de reseñas por categoría de producto:

```python
english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# Mostrar conteos para los 20 productos principales
english_df["product_category"].value_counts()[:20]
```

```python out
home                      17679
apparel                   15951
wireless                  15717
other                     13418
beauty                    12091
drugstore                 11730
kitchen                   10382
toy                        8745
sports                     8277
automotive                 7506
lawn_and_garden            7327
home_improvement           7136
pet_products               7082
digital_ebook_purchase     6749
pc                         6401
electronics                6186
office_product             5521
shoes                      5197
grocery                    4730
book                       3756
Name: product_category, dtype: int64
```

Los productos más populares en el dataset en inglés son artículos del hogar, ropa y electrónicos inalámbricos. Sin embargo, para mantener el tema de Amazon, enfoquémonos en resumir reseñas de libros -- ¡después de todo, esto es en lo que se fundó la empresa! Podemos ver dos categorías de productos que encajan (`book` y `digital_ebook_purchase`), así que filtremos los datasets en ambos idiomas solo para estos productos. Como vimos en el [Capítulo 5](/course/chapter5), la función `Dataset.filter()` nos permite segmentar un dataset de manera muy eficiente, por lo que podemos definir una función simple para hacer esto:

```python
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
```

Ahora cuando apliquemos esta función a `english_dataset` y `spanish_dataset`, el resultado contendrá solo aquellas filas que involucran las categorías de libros. Antes de aplicar el filtro, cambiemos el formato de `english_dataset` de `"pandas"` de vuelta a `"arrow"`:

```python
english_dataset.reset_format()
```

Luego podemos aplicar la función de filtro, y como verificación de cordura veamos una muestra de reseñas para ver si realmente son sobre libros:

```python
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)
```

```python out
'>> Title: I\'m dissapointed.'
'>> Review: I guess I had higher expectations for this book from the reviews. I really thought I\'d at least like it. The plot idea was great. I loved Ash but, it just didnt go anywhere. Most of the book was about their radio show and talking to callers. I wanted the author to dig deeper so we could really get to know the characters. All we know about Grace is that she is attractive looking, Latino and is kind of a brat. I\'m dissapointed.'

'>> Title: Good art, good price, poor design'
'>> Review: I had gotten the DC Vintage calendar the past two years, but it was on backorder forever this year and I saw they had shrunk the dimensions for no good reason. This one has good art choices but the design has the fold going through the picture, so it\'s less aesthetically pleasing, especially if you want to keep a picture to hang. For the price, a good calendar'

'>> Title: Helpful'
'>> Review: Nearly all the tips useful and. I consider myself an intermediate to advanced user of OneNote. I would highly recommend.'
```

De acuerdo, podemos ver que las reseñas no son estrictamente sobre libros y pueden referirse a cosas como calendarios y aplicaciones electrónicas como OneNote. Sin embargo, el dominio parece adecuado para entrenar un modelo de resúmenes. Antes de ver varios modelos que son adecuados para esta tarea, tenemos una última preparación de datos que hacer: combinar las reseñas en inglés y español en un solo objeto `DatasetDict`. 🤗 Datasets proporciona una función práctica `concatenate_datasets()` que (como sugiere el nombre) apilará dos objetos `Dataset` uno encima del otro. Entonces, para crear nuestro dataset bilingüe, recorreremos cada división, concatenaremos los datasets para esa división y mezclaremos el resultado para asegurar que nuestro modelo no se sobreajuste a un solo idioma:

```python
from datasets import concatenate_datasets, DatasetDict

books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# Echar un vistazo a algunos ejemplos
show_samples(books_dataset)
```

```python out
'>> Title: Easy to follow!!!!'
'>> Review: I loved The dash diet weight loss Solution. Never hungry. I would recommend this diet. Also the menus are well rounded. Try it. Has lots of the information need thanks.'

'>> Title: PARCIALMENTE DAÑADO'
'>> Review: Me llegó el día que tocaba, junto a otros libros que pedí, pero la caja llegó en mal estado lo cual dañó las esquinas de los libros porque venían sin protección (forro).'

'>> Title: no lo he podido descargar'
'>> Review: igual que el anterior'
```

¡Esto ciertamente parece una mezcla de reseñas en inglés y español! Ahora que tenemos un corpus de entrenamiento, una última cosa a verificar es la distribución de palabras en las reseñas y sus títulos. Esto es especialmente importante para tareas de resúmenes, donde resúmenes de referencia cortos en los datos pueden sesgar al modelo a solo generar una o dos palabras en los resúmenes generados. Los gráficos a continuación muestran las distribuciones de palabras, y podemos ver que los títulos están muy sesgados hacia solo 1-2 palabras:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/review-lengths.svg" alt="Word count distributions for the review titles and texts."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/review-lengths-dark.svg" alt="Word count distributions for the review titles and texts."/>
</div>

Para lidiar con esto, filtraremos los ejemplos con títulos muy cortos para que nuestro modelo pueda producir resúmenes más interesantes. Como estamos tratando con textos en inglés y español, podemos usar una heurística aproximada para dividir los títulos por espacios en blanco y luego usar nuestro confiable método `Dataset.filter()` de la siguiente manera:

```python
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
```

Ahora que hemos preparado nuestro corpus, ¡veamos algunos posibles modelos Transformer que uno podría ajustar finamente en él!

## Modelos para resumen de texto[[models-for-text-summarization]]

Si lo piensas, el resumen de texto es un tipo de tarea similar a la traducción automática: tenemos un cuerpo de texto como una reseña que nos gustaría "traducir" a una versión más corta que capture las características sobresalientes de la entrada. En consecuencia, la mayoría de los modelos Transformer para resúmenes adoptan la arquitectura codificador-decodificador que encontramos por primera vez en el [Capítulo 1](/course/chapter1), aunque hay algunas excepciones como la familia de modelos GPT que también se pueden usar para resúmenes en configuraciones de pocos ejemplos. La siguiente tabla lista algunos modelos preentrenados populares que pueden ser ajustados finamente para resúmenes.

| Modelo Transformer | Descripción                                                                                                                                                                                                    | ¿Multilingüe? |
| :---------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: |
|    [GPT-2](https://huggingface.co/gpt2-xl)    | Aunque fue entrenado como un modelo de lenguaje auto-regresivo, puedes hacer que GPT-2 genere resúmenes añadiendo "TL;DR" al final del texto de entrada.                                                                          |      ❌       |
|   [PEGASUS](https://huggingface.co/google/pegasus-large)   | Usa un objetivo de preentrenamiento para predecir oraciones enmascaradas en textos de múltiples oraciones. Este objetivo de preentrenamiento está más cerca de los resúmenes que el modelado de lenguaje estándar y obtiene puntuaciones altas en benchmarks populares. |      ❌       |
|     [T5](https://huggingface.co/t5-base)      | Una arquitectura Transformer universal que formula todas las tareas en un marco de texto a texto; por ejemplo, el formato de entrada para que el modelo resuma un documento es `summarize: ARTICLE`.                              |      ❌       |
|     [mT5](https://huggingface.co/google/mt5-base)     | Una versión multilingüe de T5, preentrenada en el corpus multilingüe Common Crawl (mC4), cubriendo 101 idiomas.                                                                                                |      ✅       |
|    [BART](https://huggingface.co/facebook/bart-base)     | Una arquitectura Transformer novedosa con tanto una pila de codificador como de decodificador entrenada para reconstruir entrada corrupta que combina los esquemas de preentrenamiento de BERT y GPT-2.                                    |      ❌       |
|  [mBART-50](https://huggingface.co/facebook/mbart-large-50)   | Una versión multilingüe de BART, preentrenada en 50 idiomas.                                                                                                                                                     |      ✅       |

Como puedes ver en esta tabla, la mayoría de los modelos Transformer para resúmenes (y de hecho la mayoría de las tareas de PLN) son monolingües. Esto es genial si tu tarea está en un idioma de "altos recursos" como inglés o alemán, pero no tanto para los miles de otros idiomas en uso alrededor del mundo. Afortunadamente, hay una clase de modelos Transformer multilingües, como mT5 y mBART, que vienen al rescate. Estos modelos se preentrenan usando modelado de lenguaje, pero con un giro: en lugar de entrenar en un corpus de un solo idioma, se entrenan conjuntamente en textos de más de 50 idiomas a la vez.

Nos enfocaremos en mT5, una arquitectura interesante basada en T5 que fue preentrenada en un marco de texto a texto. En T5, cada tarea de PLN se formula en términos de un prefijo de prompt como `summarize:` que condiciona al modelo para adaptar el texto generado al prompt. Como se muestra en la figura a continuación, ¡esto hace que T5 sea extremadamente versátil, ya que puedes resolver muchas tareas con un solo modelo!

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/t5.svg" alt="Different tasks performed by the T5 architecture."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/t5-dark.svg" alt="Different tasks performed by the T5 architecture."/>
</div>

mT5 no usa prefijos, pero comparte gran parte de la versatilidad de T5 y tiene la ventaja de ser multilingüe. Ahora que hemos elegido un modelo, veamos cómo preparar nuestros datos para el entrenamiento.


> [!TIP]
> **¡Pruébalo!** Una vez que hayas trabajado a través de esta sección, ve qué tan bien se compara mT5 con mBART ajustando finamente este último con las mismas técnicas. Para puntos extra, también puedes intentar ajustar finamente T5 solo en las reseñas en inglés. Como T5 tiene un prefijo de prompt especial, necesitarás anteponer `summarize:` a los ejemplos de entrada en los pasos de preprocesamiento a continuación.

## Preprocesamiento de los datos[[preprocessing-the-data]]


**Video:** [Ver en YouTube](https://youtu.be/1m7BerpSq8A)


Nuestra siguiente tarea es tokenizar y codificar nuestras reseñas y sus títulos. Como de costumbre, comenzamos cargando el tokenizador asociado con el checkpoint del modelo preentrenado. Usaremos `mt5-small` como nuestro checkpoint para poder ajustar finamente el modelo en una cantidad razonable de tiempo:

```python
from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

> [!TIP]
> En las primeras etapas de tus proyectos de PLN, una buena práctica es entrenar una clase de modelos "pequeños" en una pequeña muestra de datos. Esto te permite depurar e iterar más rápido hacia un flujo de trabajo de extremo a extremo. Una vez que estés seguro de los resultados, siempre puedes escalar el modelo simplemente cambiando el checkpoint del modelo.

Probemos el tokenizador de mT5 en un pequeño ejemplo:

```python
inputs = tokenizer("I loved reading the Hunger Games!")
inputs
```

```python out
{'input_ids': [336, 259, 28387, 11807, 287, 62893, 295, 12507, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Aquí podemos ver los familiares `input_ids` y `attention_mask` que encontramos en nuestros primeros experimentos de ajuste fino en el [Capítulo 3](/course/chapter3). Decodifiquemos estos IDs de entrada con la función `convert_ids_to_tokens()` del tokenizador para ver con qué tipo de tokenizador estamos tratando:

```python
tokenizer.convert_ids_to_tokens(inputs.input_ids)
```

```python out
['▁I', '▁', 'loved', '▁reading', '▁the', '▁Hung', 'er', '▁Games', '</s>']
```

El carácter Unicode especial `▁` y el token de fin de secuencia `</s>` indican que estamos tratando con el tokenizador SentencePiece, que está basado en el algoritmo de segmentación Unigram discutido en el [Capítulo 6](/course/chapter6). Unigram es especialmente útil para corpus multilingües ya que permite que SentencePiece sea agnóstico respecto a acentos, puntuación y el hecho de que muchos idiomas, como el japonés, no tienen caracteres de espacio en blanco.

Para tokenizar nuestro corpus, tenemos que lidiar con una sutileza asociada con los resúmenes: porque nuestras etiquetas también son texto, es posible que excedan el tamaño máximo de contexto del modelo. Esto significa que necesitamos aplicar truncamiento tanto a las reseñas como a sus títulos para asegurarnos de no pasar entradas excesivamente largas a nuestro modelo. Los tokenizadores en 🤗 Transformers proporcionan un ingenioso argumento `text_target` que te permite tokenizar las etiquetas en paralelo a las entradas. Aquí hay un ejemplo de cómo se procesan las entradas y objetivos para mT5:

```python
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Recorramos este código para entender qué está pasando. Lo primero que hemos hecho es definir valores para `max_input_length` y `max_target_length`, que establecen los límites superiores de cuán largas pueden ser nuestras reseñas y títulos. Como el cuerpo de la reseña es típicamente mucho más grande que el título, hemos escalado estos valores en consecuencia.

Con `preprocess_function()`, es entonces una simple cuestión de tokenizar todo el corpus usando la práctica función `Dataset.map()` que hemos usado extensamente a lo largo de este curso:

```python
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
```

Ahora que el corpus ha sido preprocesado, veamos algunas métricas que se usan comúnmente para resúmenes. Como veremos, no hay una solución mágica cuando se trata de medir la calidad del texto generado por máquinas.

> [!TIP]
> Quizás hayas notado que usamos `batched=True` en nuestra función `Dataset.map()` arriba. Esto codifica los ejemplos en lotes de 1,000 (el valor predeterminado) y te permite aprovechar las capacidades de multiprocesamiento de los tokenizadores rápidos en 🤗 Transformers. Donde sea posible, intenta usar `batched=True` para sacar el máximo provecho de tu preprocesamiento.


## Métricas para resumen de texto[[metrics-for-text-summarization]]


**Video:** [Ver en YouTube](https://youtu.be/TMshhnrEXlg)


En comparación con la mayoría de las otras tareas que hemos cubierto en este curso, medir el rendimiento de tareas de generación de texto como resúmenes o traducción no es tan sencillo. Por ejemplo, dada una reseña como "I loved reading the Hunger Games", hay múltiples resúmenes válidos, como "I loved the Hunger Games" o "Hunger Games is a great read". Claramente, aplicar algún tipo de coincidencia exacta entre el resumen generado y la etiqueta no es una buena solución -- incluso los humanos tendrían un mal desempeño bajo tal métrica, porque todos tenemos nuestro propio estilo de escritura.

Para resúmenes, una de las métricas más comúnmente usadas es la [puntuación ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) (abreviatura de Recall-Oriented Understudy for Gisting Evaluation). La idea básica detrás de esta métrica es comparar un resumen generado contra un conjunto de resúmenes de referencia que típicamente son creados por humanos. Para hacer esto más preciso, supongamos que queremos comparar los siguientes dos resúmenes:

```python
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
```

Una forma de compararlos podría ser contar el número de palabras que se solapan, que en este caso sería 6. Sin embargo, esto es un poco burdo, así que en su lugar ROUGE se basa en calcular las puntuaciones de _precisión_ y _recall_ para el solapamiento.

> [!TIP]
> No te preocupes si esta es la primera vez que escuchas sobre precisión y recall -- repasaremos algunos ejemplos explícitos juntos para dejarlo todo claro. Estas métricas se encuentran usualmente en tareas de clasificación, así que si quieres entender cómo se definen precisión y recall en ese contexto, recomendamos revisar las [guías](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) de `scikit-learn`.

Para ROUGE, el recall mide cuánto del resumen de referencia es capturado por el generado. Si solo estamos comparando palabras, el recall se puede calcular según la siguiente fórmula:

$$ \mathrm{Recall} = \frac{\mathrm{Número\,de\,palabras\, que\,se\,solapan}}{\mathrm{Número\, total\, de\, palabras\, en\, resumen\, de\, referencia}} $$

Para nuestro ejemplo simple anterior, esta fórmula da un recall perfecto de 6/6 = 1; es decir, todas las palabras en el resumen de referencia han sido producidas por el modelo. Esto puede sonar genial, pero imagina si nuestro resumen generado hubiera sido "I really really loved reading the Hunger Games all night". Esto también tendría un recall perfecto, pero es posiblemente un peor resumen ya que es verboso. Para lidiar con estos escenarios también calculamos la precisión, que en el contexto de ROUGE mide cuánto del resumen generado fue relevante:

$$ \mathrm{Precisión} = \frac{\mathrm{Número\,de\,palabras\, que\,se\,solapan}}{\mathrm{Número\, total\, de\, palabras\, en\, resumen\, generado}} $$

Aplicando esto a nuestro resumen verboso da una precisión de 6/10 = 0.6, que es considerablemente peor que la precisión de 6/7 = 0.86 obtenida por el más corto. En la práctica, usualmente se calculan tanto precisión como recall, y luego se reporta la puntuación F1 (la media armónica de precisión y recall). Podemos hacer esto fácilmente en 🤗 Datasets instalando primero el paquete `rouge_score`:

```py
!pip install rouge_score
```

y luego cargando la métrica ROUGE de la siguiente manera:

```python
import evaluate

rouge_score = evaluate.load("rouge")
```

Luego podemos usar la función `rouge_score.compute()` para calcular todas las métricas a la vez:

```python
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
scores
```

```python out
{'rouge1': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rouge2': AggregateScore(low=Score(precision=0.67, recall=0.8, fmeasure=0.73), mid=Score(precision=0.67, recall=0.8, fmeasure=0.73), high=Score(precision=0.67, recall=0.8, fmeasure=0.73)),
 'rougeL': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rougeLsum': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92))}
```

Vaya, hay mucha información en esa salida -- ¿qué significa todo? Primero, 🤗 Datasets en realidad calcula intervalos de confianza para precisión, recall y puntuación F1; estos son los atributos `low`, `mid` y `high` que puedes ver aquí. Además, 🤗 Datasets calcula una variedad de puntuaciones ROUGE que están basadas en diferentes tipos de granularidad de texto al comparar los resúmenes generados y de referencia. La variante `rouge1` es el solapamiento de unigramas -- esto es solo una forma elegante de decir el solapamiento de palabras y es exactamente la métrica que hemos discutido arriba. Para verificar esto, extraigamos el valor `mid` de nuestras puntuaciones:

```python
scores["rouge1"].mid
```

```python out
Score(precision=0.86, recall=1.0, fmeasure=0.92)
```

¡Genial, los números de precisión y recall coinciden! Ahora, ¿qué hay de esas otras puntuaciones ROUGE? `rouge2` mide el solapamiento entre bigramas (piensa en el solapamiento de pares de palabras), mientras que `rougeL` y `rougeLsum` miden las secuencias coincidentes más largas de palabras buscando las subcadenas comunes más largas en los resúmenes generados y de referencia. El "sum" en `rougeLsum` se refiere al hecho de que esta métrica se calcula sobre un resumen completo, mientras que `rougeL` se calcula como el promedio sobre oraciones individuales.

> [!TIP]
> **¡Pruébalo!** Crea tu propio ejemplo de un resumen generado y de referencia y ve si las puntuaciones ROUGE resultantes coinciden con un cálculo manual basado en las fórmulas de precisión y recall. Para puntos extra, divide el texto en bigramas y compara la precisión y recall para la métrica `rouge2`.

Usaremos estas puntuaciones ROUGE para rastrear el rendimiento de nuestro modelo, pero antes de hacerlo hagamos algo que todo buen practicante de PLN debería hacer: ¡crear una línea base fuerte pero simple!

### Creando una línea base fuerte[[creating-a-strong-baseline]]

Una línea base común para resumen de texto es simplemente tomar las primeras tres oraciones de un artículo, a menudo llamada la línea base _lead-3_. Podríamos usar puntos finales para rastrear los límites de las oraciones, pero esto fallará en acrónimos como "U.S." o "U.N." -- así que en su lugar usaremos la biblioteca `nltk`, que incluye un mejor algoritmo para manejar estos casos. Puedes instalar el paquete usando `pip` de la siguiente manera:

```python
!pip install nltk
```

y luego descargar las reglas de puntuación:

```python
import nltk

nltk.download("punkt")
```

A continuación, importamos el tokenizador de oraciones de `nltk` y creamos una función simple para extraer las primeras tres oraciones en una reseña. La convención en resumen de texto es separar cada resumen con una nueva línea, así que también incluyamos esto y probémoslo en un ejemplo de entrenamiento:

```python
from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


print(three_sentence_summary(books_dataset["train"][1]["review_body"]))
```

```python out
'I grew up reading Koontz, and years ago, I stopped,convinced i had "outgrown" him.'
'Still,when a friend was looking for something suspenseful too read, I suggested Koontz.'
'She found Strangers.'
```

Esto parece funcionar, así que ahora implementemos una función que extrae estos "resúmenes" de un dataset y calcula las puntuaciones ROUGE para la línea base:

```python
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])
```

Luego podemos usar esta función para calcular las puntuaciones ROUGE sobre el conjunto de validación y embellecerlas un poco usando Pandas:

```python
import pandas as pd

score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
rouge_dict
```

```python out
{'rouge1': 16.74, 'rouge2': 8.83, 'rougeL': 15.6, 'rougeLsum': 15.96}
```

Podemos ver que la puntuación `rouge2` es significativamente más baja que el resto; esto probablemente refleja el hecho de que los títulos de reseñas son típicamente concisos y así la línea base lead-3 es demasiado verbosa. Ahora que tenemos una buena línea base de la cual partir, ¡centrémonos en ajustar finamente mT5!


**PyTorch:**

## Ajuste fino de mT5 con la API `Trainer`[[fine-tuning-mt5-with-the-trainer-api]]

Ajustar finamente un modelo para resúmenes es muy similar a las otras tareas que hemos cubierto en este capítulo. Lo primero que necesitamos hacer es cargar el modelo preentrenado desde el checkpoint `mt5-small`. Como los resúmenes son una tarea de secuencia a secuencia, podemos cargar el modelo con la clase `AutoModelForSeq2SeqLM`, que descargará y almacenará en caché automáticamente los pesos:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

**TensorFlow/Keras:**

## Ajuste fino de mT5 con Keras[[fine-tuning-mt5-with-keras]]

Ajustar finamente un modelo para resúmenes es muy similar a las otras tareas que hemos cubierto en este capítulo. Lo primero que necesitamos hacer es cargar el modelo preentrenado desde el checkpoint `mt5-small`. Como los resúmenes son una tarea de secuencia a secuencia, podemos cargar el modelo con la clase `TFAutoModelForSeq2SeqLM`, que descargará y almacenará en caché automáticamente los pesos:

```python
from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```


> [!TIP]
> Si te preguntas por qué no ves ninguna advertencia sobre ajustar finamente el modelo en una tarea posterior, eso es porque para tareas de secuencia a secuencia mantenemos todos los pesos de la red. Compara esto con nuestro modelo de clasificación de texto en el [Capítulo 3](/course/chapter3), donde la cabeza del modelo preentrenado fue reemplazada por una red inicializada aleatoriamente.

Lo siguiente que necesitamos hacer es iniciar sesión en el Hugging Face Hub. Si estás ejecutando este código en un notebook, puedes hacerlo con la siguiente función de utilidad:

```python
from huggingface_hub import notebook_login

notebook_login()
```

que mostrará un widget donde puedes ingresar tus credenciales. Alternativamente, puedes ejecutar este comando en tu terminal e iniciar sesión allí:

```
huggingface-cli login
```


**PyTorch:**

Necesitaremos generar resúmenes para calcular las puntuaciones ROUGE durante el entrenamiento. Afortunadamente, 🤗 Transformers proporciona clases dedicadas `Seq2SeqTrainingArguments` y `Seq2SeqTrainer` que pueden hacer esto automáticamente. Para ver cómo funciona esto, primero definamos los hiperparámetros y otros argumentos para nuestros experimentos:

```python
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# Mostrar la pérdida de entrenamiento con cada época
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)
```

Aquí, el argumento `predict_with_generate` se ha establecido para indicar que debemos generar resúmenes durante la evaluación para poder calcular las puntuaciones ROUGE para cada época. Como se discutió en el [Capítulo 1](/course/chapter1), el decodificador realiza inferencia prediciendo tokens uno por uno, y esto es implementado por el método `generate()` del modelo. Establecer `predict_with_generate=True` le dice al `Seq2SeqTrainer` que use ese método para la evaluación. También hemos ajustado algunos de los hiperparámetros predeterminados, como la tasa de aprendizaje, número de épocas y decaimiento de peso, y hemos establecido la opción `save_total_limit` para solo guardar hasta 3 checkpoints durante el entrenamiento -- esto es porque incluso la versión "small" de mT5 usa alrededor de un GB de espacio en disco duro, y podemos ahorrar un poco de espacio limitando el número de copias que guardamos.

El argumento `push_to_hub=True` nos permitirá subir el modelo al Hub después del entrenamiento; encontrarás el repositorio bajo tu perfil de usuario en la ubicación definida por `output_dir`. Ten en cuenta que puedes especificar el nombre del repositorio al que quieres subir con el argumento `hub_model_id` (en particular, tendrás que usar este argumento para subir a una organización). Por ejemplo, cuando subimos el modelo a la [organización `huggingface-course`](https://huggingface.co/huggingface-course), añadimos `hub_model_id="huggingface-course/mt5-finetuned-amazon-en-es"` a `Seq2SeqTrainingArguments`.

Lo siguiente que necesitamos hacer es proporcionar al entrenador una función `compute_metrics()` para poder evaluar nuestro modelo durante el entrenamiento. Para resúmenes esto es un poco más complicado que simplemente llamar a `rouge_score.compute()` en las predicciones del modelo, ya que necesitamos _decodificar_ las salidas y etiquetas a texto antes de poder calcular las puntuaciones ROUGE. La siguiente función hace exactamente eso, y también hace uso de la función `sent_tokenize()` de `nltk` para separar las oraciones del resumen con nuevas líneas:

```python
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decodificar resúmenes generados a texto
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Reemplazar -100 en las etiquetas ya que no podemos decodificarlos
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decodificar resúmenes de referencia a texto
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE espera una nueva línea después de cada oración
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Calcular puntuaciones ROUGE
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extraer las puntuaciones medianas
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
```


A continuación, necesitamos definir un collator de datos para nuestra tarea de secuencia a secuencia. Como mT5 es un modelo Transformer codificador-decodificador, una sutileza al preparar nuestros lotes es que durante la decodificación necesitamos desplazar las etiquetas a la derecha por uno. Esto es necesario para asegurar que el decodificador solo vea las etiquetas de verdad anteriores y no las actuales o futuras, que serían fáciles de memorizar para el modelo. Esto es similar a cómo se aplica la auto-atención enmascarada a las entradas en una tarea como el [modelado de lenguaje causal](/course/chapter7/6).

Afortunadamente, 🤗 Transformers proporciona un collator `DataCollatorForSeq2Seq` que rellenará dinámicamente las entradas y las etiquetas por nosotros. Para instanciar este collator, simplemente necesitamos proporcionar el `tokenizer` y el `model`:


**PyTorch:**

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

**TensorFlow/Keras:**

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
```


Veamos qué produce este collator cuando se le alimenta un pequeño lote de ejemplos. Primero, necesitamos eliminar las columnas con cadenas porque el collator no sabrá cómo rellenar estos elementos:

```python
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
```

Como el collator espera una lista de `dict`s, donde cada `dict` representa un solo ejemplo en el dataset, también necesitamos convertir los datos al formato esperado antes de pasarlos al collator de datos:

```python
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
```

```python out
{'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'input_ids': tensor([[  1494,    259,   8622,    390,    259,    262,   2316,   3435,    955,
            772,    281,    772,   1617,    263,    305,  14701,    260,   1385,
           3031,    259,  24146,    332,   1037,    259,  43906,    305,    336,
            260,      1,      0,      0,      0,      0,      0,      0],
        [   259,  27531,  13483,    259,   7505,    260, 112240,  15192,    305,
          53198,    276,    259,  74060,    263,    260,    459,  25640,    776,
           2119,    336,    259,   2220,    259,  18896,    288,   4906,    288,
           1037,   3931,    260,   7083, 101476,   1143,    260,      1]]), 'labels': tensor([[ 7483,   259,  2364, 15695,     1,  -100],
        [  259, 27531, 13483,   259,  7505,     1]]), 'decoder_input_ids': tensor([[    0,  7483,   259,  2364, 15695,     1],
        [    0,   259, 27531, 13483,   259,  7505]])}
```

Lo principal a notar aquí es que el primer ejemplo es más largo que el segundo, por lo que los `input_ids` y `attention_mask` del segundo ejemplo han sido rellenados a la derecha con un token `[PAD]` (cuyo ID es `0`). De manera similar, podemos ver que las `labels` han sido rellenadas con `-100`s, para asegurar que los tokens de relleno sean ignorados por la función de pérdida. Y finalmente, podemos ver un nuevo `decoder_input_ids` que ha desplazado las etiquetas a la derecha insertando un token `[PAD]` en la primera entrada.


**PyTorch:**

¡Finalmente tenemos todos los ingredientes que necesitamos para entrenar! Ahora simplemente necesitamos instanciar el entrenador con los argumentos estándar:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

y lanzar nuestra ejecución de entrenamiento:

```python
trainer.train()
```

Durante el entrenamiento, deberías ver la pérdida de entrenamiento disminuir y las puntuaciones ROUGE aumentar con cada época. Una vez que el entrenamiento esté completo, puedes ver las puntuaciones ROUGE finales ejecutando `Trainer.evaluate()`:

```python
trainer.evaluate()
```

```python out
{'eval_loss': 3.028524398803711,
 'eval_rouge1': 16.9728,
 'eval_rouge2': 8.2969,
 'eval_rougeL': 16.8366,
 'eval_rougeLsum': 16.851,
 'eval_gen_len': 10.1597,
 'eval_runtime': 6.1054,
 'eval_samples_per_second': 38.982,
 'eval_steps_per_second': 4.914}
```

De las puntuaciones podemos ver que nuestro modelo ha superado fácilmente nuestra línea base lead-3 -- ¡bien! Lo último que queda por hacer es subir los pesos del modelo al Hub, de la siguiente manera:

```
trainer.push_to_hub(commit_message="Training complete", tags="summarization")
```

```python out
'https://huggingface.co/huggingface-course/mt5-finetuned-amazon-en-es/commit/aa0536b829b28e73e1e4b94b8a5aacec420d40e0'
```

Esto guardará el checkpoint y los archivos de configuración en `output_dir`, antes de subir todos los archivos al Hub. Al especificar el argumento `tags`, también nos aseguramos de que el widget en el Hub sea uno para un pipeline de resúmenes en lugar del de generación de texto predeterminado asociado con la arquitectura mT5 (para más información sobre etiquetas de modelos, consulta la [documentación del 🤗 Hub](https://huggingface.co/docs/hub/main#how-is-a-models-type-of-inference-api-and-widget-determined)). La salida de `trainer.push_to_hub()` es una URL al hash del commit de Git, ¡así que puedes ver fácilmente los cambios que se hicieron al repositorio del modelo!

Para cerrar esta sección, veamos cómo también podemos ajustar finamente mT5 usando las características de bajo nivel proporcionadas por 🤗 Accelerate.

**TensorFlow/Keras:**

¡Ya casi estamos listos para entrenar! Solo necesitamos convertir nuestros datasets a `tf.data.Dataset`s usando el collator de datos que definimos arriba, y luego `compile()` y `fit()` el modelo. Primero, los datasets:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=8,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=8,
)
```

Ahora, definimos nuestros hiperparámetros de entrenamiento y compilamos:

```python
from transformers import create_optimizer
import tensorflow as tf

# El número de pasos de entrenamiento es el número de muestras en el dataset, dividido por el tamaño del lote y luego multiplicado
# por el número total de épocas. Ten en cuenta que el tf_train_dataset aquí es un tf.data.Dataset por lotes,
# no el Dataset original de Hugging Face, por lo que su len() ya es num_samples // batch_size.
num_train_epochs = 8
num_train_steps = len(tf_train_dataset) * num_train_epochs
model_name = model_checkpoint.split("/")[-1]

optimizer, schedule = create_optimizer(
    init_lr=5.6e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

model.compile(optimizer=optimizer)

# Entrenar en precisión mixta float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Y finalmente, ajustamos el modelo. Usamos un `PushToHubCallback` para guardar el modelo en el Hub después de cada época, lo que nos permitirá usarlo para inferencia más tarde:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(
    output_dir=f"{model_name}-finetuned-amazon-en-es", tokenizer=tokenizer
)

model.fit(
    tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback], epochs=8
)
```

Obtuvimos algunos valores de pérdida durante el entrenamiento, pero realmente nos gustaría ver las métricas ROUGE que calculamos antes. Para obtener esas métricas, necesitaremos generar salidas del modelo y convertirlas a cadenas. Construyamos algunas listas de etiquetas y predicciones para que la métrica ROUGE las compare (ten en cuenta que si obtienes errores de importación para esta sección, puede que necesites `!pip install tqdm`). También usaremos un truco que mejora dramáticamente el rendimiento - compilando nuestro código de generación con [XLA](https://www.tensorflow.org/xla), el compilador de álgebra lineal acelerada de TensorFlow. XLA aplica varias optimizaciones al grafo de cálculo del modelo, y resulta en mejoras significativas en velocidad y uso de memoria. Como se describe en el [blog](https://huggingface.co/blog/tf-xla-generate) de Hugging Face, XLA funciona mejor cuando nuestras formas de entrada no varían demasiado. Para manejar esto, rellenaremos nuestras entradas a múltiplos de 128, y haremos un nuevo dataset con el collator de relleno, y luego aplicaremos el decorador `@tf.function(jit_compile=True)` a nuestra función de generación, que marca toda la función para compilación con XLA.

```python
from tqdm import tqdm
import numpy as np

generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=320
)

tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
    drop_remainder=True,
)


@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=32,
    )


all_preds = []
all_labels = []
for batch, labels in tqdm(tf_generate_dataset):
    predictions = generate_with_xla(batch)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = labels.numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    all_preds.extend(decoded_preds)
    all_labels.extend(decoded_labels)
```

Una vez que tenemos nuestras listas de cadenas de etiquetas y predicciones, calcular la puntuación ROUGE es fácil:

```python
result = rouge_score.compute(
    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
{k: round(v, 4) for k, v in result.items()}
```

```
{'rouge1': 31.4815, 'rouge2': 25.4386, 'rougeL': 31.4815, 'rougeLsum': 31.4815}
```


**PyTorch:**

## Ajuste fino de mT5 con 🤗 Accelerate[[fine-tuning-mt5-with-accelerate]]

Ajustar finamente nuestro modelo con 🤗 Accelerate es muy similar al ejemplo de clasificación de texto que encontramos en el [Capítulo 3](/course/chapter3). Las principales diferencias serán la necesidad de generar explícitamente nuestros resúmenes durante el entrenamiento y definir cómo calculamos las puntuaciones ROUGE (recuerda que el `Seq2SeqTrainer` se encargó de la generación por nosotros). ¡Veamos cómo podemos implementar estos dos requisitos dentro de 🤗 Accelerate!

### Preparando todo para el entrenamiento[[preparing-everything-for-training]]

Lo primero que necesitamos hacer es crear un `DataLoader` para cada una de nuestras divisiones. Como los dataloaders de PyTorch esperan lotes de tensores, necesitamos establecer el formato a `"torch"` en nuestros datasets:

```python
tokenized_datasets.set_format("torch")
```

Ahora que tenemos datasets que consisten solo en tensores, lo siguiente que hay que hacer es instanciar el `DataCollatorForSeq2Seq` de nuevo. Para esto necesitamos proporcionar una versión fresca del modelo, así que carguémoslo de nuevo desde nuestra caché:

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

Luego podemos instanciar el collator de datos y usar esto para definir nuestros dataloaders:

```python
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
)
```

Lo siguiente que hay que hacer es definir el optimizador que queremos usar. Como en nuestros otros ejemplos, usaremos `AdamW`, que funciona bien para la mayoría de los problemas:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Finalmente, alimentamos nuestro modelo, optimizador y dataloaders al método `accelerator.prepare()`:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> Si estás entrenando en una TPU, necesitarás mover todo el código anterior a una función de entrenamiento dedicada. Consulta el [Capítulo 3](/course/chapter3) para más detalles.

Ahora que hemos preparado nuestros objetos, quedan tres cosas por hacer:

* Definir el programa de tasa de aprendizaje.
* Implementar una función para post-procesar los resúmenes para evaluación.
* Crear un repositorio en el Hub al que podamos subir nuestro modelo.

Para el programa de tasa de aprendizaje, usaremos el lineal estándar de secciones anteriores:

```python
from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Para el post-procesamiento, necesitamos una función que divida los resúmenes generados en oraciones separadas por nuevas líneas. Este es el formato que espera la métrica ROUGE, y podemos lograrlo con el siguiente fragmento de código:

```python
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE espera una nueva línea después de cada oración
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
```

Esto debería resultarte familiar si recuerdas cómo definimos la función `compute_metrics()` del `Seq2SeqTrainer`.

Finalmente, necesitamos crear un repositorio de modelo en el Hugging Face Hub. Para esto, podemos usar la biblioteca 🤗 Hub apropiadamente titulada. Solo necesitamos definir un nombre para nuestro repositorio, y la biblioteca tiene una función de utilidad para combinar el ID del repositorio con el perfil del usuario:

```python
from huggingface_hub import get_full_repo_name

model_name = "test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'lewtun/mt5-finetuned-amazon-en-es-accelerate'
```

Ahora podemos usar este nombre de repositorio para clonar una versión local a nuestro directorio de resultados que almacenará los artefactos de entrenamiento:

```python
from huggingface_hub import Repository

output_dir = "results-mt5-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

¡Esto nos permitirá subir los artefactos de vuelta al Hub llamando al método `repo.push_to_hub()` durante el entrenamiento! Ahora concluyamos nuestro análisis escribiendo el bucle de entrenamiento.

### Bucle de entrenamiento[[training-loop]]

El bucle de entrenamiento para resúmenes es bastante similar a los otros ejemplos de 🤗 Accelerate que hemos encontrado y se divide aproximadamente en cuatro pasos principales:

1. Entrenar el modelo iterando sobre todos los ejemplos en `train_dataloader` para cada época.
2. Generar resúmenes del modelo al final de cada época, primero generando los tokens y luego decodificándolos (y los resúmenes de referencia) a texto.
3. Calcular las puntuaciones ROUGE usando las mismas técnicas que vimos antes.
4. Guardar los checkpoints y subir todo al Hub. Aquí confiamos en el ingenioso argumento `blocking=False` del objeto `Repository` para que podamos subir los checkpoints por época _asincrónicamente_. Esto nos permite continuar entrenando sin tener que esperar la subida algo lenta asociada con un modelo de un GB de tamaño.

Estos pasos se pueden ver en el siguiente bloque de código:

```python
from tqdm.auto import tqdm
import torch
import numpy as np

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Entrenamiento
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluación
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # Si no rellenamos a longitud máxima, también necesitamos rellenar las etiquetas
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Reemplazar -100 en las etiquetas ya que no podemos decodificarlos
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Calcular métricas
    result = rouge_score.compute()
    # Extraer las puntuaciones ROUGE medianas
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Guardar y subir
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
```

```python out
Epoch 0: {'rouge1': 5.6351, 'rouge2': 1.1625, 'rougeL': 5.4866, 'rougeLsum': 5.5005}
Epoch 1: {'rouge1': 9.8646, 'rouge2': 3.4106, 'rougeL': 9.9439, 'rougeLsum': 9.9306}
Epoch 2: {'rouge1': 11.0872, 'rouge2': 3.3273, 'rougeL': 11.0508, 'rougeLsum': 10.9468}
Epoch 3: {'rouge1': 11.8587, 'rouge2': 4.8167, 'rougeL': 11.7986, 'rougeLsum': 11.7518}
Epoch 4: {'rouge1': 12.9842, 'rouge2': 5.5887, 'rougeL': 12.7546, 'rougeLsum': 12.7029}
Epoch 5: {'rouge1': 13.4628, 'rouge2': 6.4598, 'rougeL': 13.312, 'rougeLsum': 13.2913}
Epoch 6: {'rouge1': 12.9131, 'rouge2': 5.8914, 'rougeL': 12.6896, 'rougeLsum': 12.5701}
Epoch 7: {'rouge1': 13.3079, 'rouge2': 6.2994, 'rougeL': 13.1536, 'rougeLsum': 13.1194}
Epoch 8: {'rouge1': 13.96, 'rouge2': 6.5998, 'rougeL': 13.9123, 'rougeLsum': 13.7744}
Epoch 9: {'rouge1': 14.1192, 'rouge2': 7.0059, 'rougeL': 14.1172, 'rougeLsum': 13.9509}
```

¡Y eso es todo! Una vez que ejecutes esto, tendrás un modelo y resultados que son bastante similares a los que obtuvimos con el `Trainer`.


## Usando tu modelo ajustado finamente[[using-your-fine-tuned-model]]

Una vez que hayas subido el modelo al Hub, puedes jugar con él ya sea a través del widget de inferencia o con un objeto `pipeline`, de la siguiente manera:

```python
from transformers import pipeline

hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)
```

Podemos alimentar algunos ejemplos del conjunto de prueba (que el modelo no ha visto) a nuestro pipeline para tener una idea de la calidad de los resúmenes. Primero implementemos una función simple para mostrar la reseña, el título y el resumen generado juntos:

```python
def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")
```

Veamos uno de los ejemplos en inglés que obtenemos:

```python
print_summary(100)
```

```python out
'>>> Review: Nothing special at all about this product... the book is too small and stiff and hard to write in. The huge sticker on the back doesn't come off and looks super tacky. I would not purchase this again. I could have just bought a journal from the dollar store and it would be basically the same thing. It's also really expensive for what it is.'

'>>> Title: Not impressed at all... buy something else'

'>>> Summary: Nothing special at all about this product'
```

¡Esto no está nada mal! Podemos ver que nuestro modelo realmente ha sido capaz de realizar resúmenes _abstractivos_ aumentando partes de la reseña con palabras nuevas. Y quizás el aspecto más genial de nuestro modelo es que es bilingüe, así que también podemos generar resúmenes de reseñas en español:

```python
print_summary(0)
```

```python out
'>>> Review: Es una trilogia que se hace muy facil de leer. Me ha gustado, no me esperaba el final para nada'

'>>> Title: Buena literatura para adolescentes'

'>>> Summary: Muy facil de leer'
```

El resumen se traduce como "Muy fácil de leer" en inglés, que podemos ver en este caso fue extraído directamente de la reseña. Sin embargo, esto muestra la versatilidad del modelo mT5 y te ha dado una muestra de lo que es lidiar con un corpus multilingüe.

A continuación, centraremos nuestra atención en una tarea un poco más compleja: entrenar un modelo de lenguaje desde cero.


---



# Entrenamiento de un modelo de lenguaje causal desde cero[[training-a-causal-language-model-from-scratch]]


Hasta ahora, hemos utilizado principalmente modelos preentrenados y los hemos ajustado finamente para nuevos casos de uso reutilizando los pesos del preentrenamiento. Como vimos en el [Capítulo 1](/course/chapter1), esto se conoce comúnmente como _aprendizaje por transferencia_, y es una estrategia muy exitosa para aplicar modelos Transformer a la mayoría de casos de uso del mundo real donde los datos etiquetados son escasos. En este capítulo, tomaremos un enfoque diferente y entrenaremos un modelo completamente nuevo desde cero. Este es un buen enfoque si tienes muchos datos y estos son muy diferentes de los datos de preentrenamiento utilizados para los modelos disponibles. Sin embargo, también requiere considerablemente más recursos computacionales para preentrenar un modelo de lenguaje que simplemente ajustar finamente uno existente. Ejemplos donde puede tener sentido entrenar un nuevo modelo incluyen conjuntos de datos que consisten en notas musicales, secuencias moleculares como ADN, o lenguajes de programación. Estos últimos han ganado tracción recientemente gracias a herramientas como TabNine y Copilot de GitHub, impulsadas por el modelo Codex de OpenAI, que pueden generar largas secuencias de código. Esta tarea de generación de texto se aborda mejor con modelos auto-regresivos o de modelado de lenguaje causal como GPT-2.

En esta sección construiremos una versión reducida de un modelo de generación de código: nos enfocaremos en completar líneas individuales en lugar de funciones o clases completas, utilizando un subconjunto de código Python. Cuando trabajas con datos en Python, estás en contacto frecuente con el stack de ciencia de datos de Python, que consiste en las bibliotecas `matplotlib`, `seaborn`, `pandas` y `scikit-learn`. Al usar estos frameworks, es común necesitar buscar comandos específicos, así que sería útil si pudiéramos usar un modelo para completar estas llamadas por nosotros.


**Video:** [Ver en YouTube](https://youtu.be/Vpjb1lu0MDk)


En el [Capítulo 6](/course/chapter6) creamos un tokenizador eficiente para procesar código fuente de Python, pero lo que aún necesitamos es un conjunto de datos a gran escala para preentrenar un modelo. Aquí, aplicaremos nuestro tokenizador a un corpus de código Python derivado de repositorios de GitHub. Luego usaremos la API `Trainer` y 🤗 Accelerate para entrenar el modelo. ¡Empecemos!

<iframe src="https://course-demos-codeparrot-ds.hf.space" frameBorder="0" height="300" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

Esto muestra el modelo que fue entrenado y subido al Hub usando el código mostrado en esta sección. Puedes encontrarlo [aquí](https://huggingface.co/huggingface-course/codeparrot-ds?text=plt.imshow%28). Ten en cuenta que debido a la aleatoriedad en la generación de texto, probablemente obtendrás un resultado ligeramente diferente.

## Recopilando los datos[[gathering-the-data]]

El código Python está abundantemente disponible en repositorios de código como GitHub, que podemos usar para crear un conjunto de datos extrayendo cada repositorio de Python. Este fue el enfoque tomado en el [libro de texto de Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/) para preentrenar un modelo GPT-2 grande. Usando un volcado de GitHub de aproximadamente 180 GB que contiene alrededor de 20 millones de archivos Python llamado `codeparrot`, los autores construyeron un conjunto de datos que luego compartieron en el [Hugging Face Hub](https://huggingface.co/datasets/transformersbook/codeparrot).

Sin embargo, entrenar con el corpus completo consume mucho tiempo y recursos computacionales, y solo necesitamos el subconjunto del conjunto de datos relacionado con el stack de ciencia de datos de Python. Así que, comencemos filtrando el conjunto de datos `codeparrot` para todos los archivos que incluyan cualquiera de las bibliotecas en este stack. Debido al tamaño del conjunto de datos, queremos evitar descargarlo; en su lugar, usaremos la función de streaming para filtrarlo sobre la marcha. Para ayudarnos a filtrar las muestras de código usando las bibliotecas que mencionamos anteriormente, usaremos la siguiente función:

```py
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False
```

Probémosla con dos ejemplos:

```py
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"

print(
    any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters)
)
```

```python out
False True
```

Podemos usar esto para crear una función que transmitirá el conjunto de datos y filtrará los elementos que queremos:

```py
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)
```

Luego simplemente podemos aplicar esta función al conjunto de datos en streaming:

```py
# Esta celda tardará mucho tiempo en ejecutarse, así que deberías saltarla e ir a
# la siguiente.
from datasets import load_dataset

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)
```

```python out
3.26% of data after filtering.
```

Esto nos deja con aproximadamente el 3% del conjunto de datos original, que sigue siendo bastante considerable -- el conjunto de datos resultante es de 6 GB y consiste en 600,000 scripts de Python.

Filtrar el conjunto de datos completo puede tomar 2-3 horas dependiendo de tu máquina y ancho de banda. Si no quieres pasar por este largo proceso tú mismo, proporcionamos el conjunto de datos filtrado en el Hub para que lo descargues:

```py
from datasets import load_dataset, DatasetDict

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 606720
    })
    valid: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 3322
    })
})
```

> [!TIP]
> El preentrenamiento del modelo de lenguaje tomará un tiempo. Sugerimos que primero ejecutes el bucle de entrenamiento en una muestra de los datos descomentando las dos líneas parciales de arriba, y asegúrate de que el entrenamiento se complete exitosamente y los modelos se almacenen. Nada es más frustrante que una ejecución de entrenamiento que falla en el último paso porque olvidaste crear una carpeta o porque hay un error tipográfico al final del bucle de entrenamiento.

Veamos un ejemplo del conjunto de datos. Solo mostraremos los primeros 200 caracteres de cada campo:

```py
for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")
```

```python out
'REPO_NAME: kmike/scikit-learn'
'PATH: sklearn/utils/__init__.py'
'COPIES: 3'
'SIZE: 10094'
'''CONTENT: """
The :mod:`sklearn.utils` module includes various utilites.
"""

from collections import Sequence

import numpy as np
from scipy.sparse import issparse
import warnings

from .murmurhash import murm
LICENSE: bsd-3-clause'''
```

Podemos ver que el campo `content` contiene el código en el que queremos que nuestro modelo se entrene. Ahora que tenemos un conjunto de datos, necesitamos preparar los textos para que estén en un formato adecuado para el preentrenamiento.

## Preparando el conjunto de datos[[preparing-the-dataset]]


**Video:** [Ver en YouTube](https://youtu.be/ma1TrR7gE7I)


El primer paso será tokenizar los datos, para poder usarlos en el entrenamiento. Como nuestro objetivo es principalmente autocompletar llamadas cortas a funciones, podemos mantener el tamaño del contexto relativamente pequeño. Esto tiene el beneficio de que podemos entrenar el modelo mucho más rápido y requiere significativamente menos memoria. Si es importante para tu aplicación tener más contexto (por ejemplo, si quieres que el modelo escriba pruebas unitarias basadas en un archivo con la definición de la función), asegúrate de aumentar ese número, pero también ten en cuenta que esto viene con una mayor huella de memoria de GPU. Por ahora, fijemos el tamaño del contexto en 128 tokens, a diferencia de los 1,024 o 2,048 usados en GPT-2 o GPT-3, respectivamente.

La mayoría de los documentos contienen muchos más de 128 tokens, así que simplemente truncar las entradas a la longitud máxima eliminaría una gran fracción de nuestro conjunto de datos. En su lugar, usaremos la opción `return_overflowing_tokens` para tokenizar toda la entrada y dividirla en varios fragmentos, como hicimos en el [Capítulo 6](/course/chapter6/4). También usaremos la opción `return_length` para devolver automáticamente la longitud de cada fragmento creado. A menudo el último fragmento será más pequeño que el tamaño del contexto, y nos desharemos de estas piezas para evitar problemas de padding; realmente no las necesitamos ya que tenemos muchos datos de todos modos.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/chunking_texts.svg" alt="Dividiendo un texto grande en varios fragmentos."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/chunking_texts-dark.svg" alt="Dividiendo un texto grande en varios fragmentos."/>
</div>

Veamos exactamente cómo funciona esto mirando los primeros dos ejemplos:

```py
from transformers import AutoTokenizer

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
```

```python out
Input IDs length: 34
Input chunk lengths: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 117, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 41]
Chunk mapping: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Podemos ver que obtenemos 34 segmentos en total de esos dos ejemplos. Mirando las longitudes de los fragmentos, podemos ver que los fragmentos al final de ambos documentos tienen menos de 128 tokens (117 y 41, respectivamente). Estos representan solo una pequeña fracción del total de fragmentos que tenemos, así que podemos descartarlos de forma segura. Con el campo `overflow_to_sample_mapping`, también podemos reconstruir qué fragmentos pertenecían a qué muestras de entrada.

Con esta operación estamos usando una característica útil de la función `Dataset.map()` en 🤗 Datasets, que es que no requiere mapeos uno a uno; como vimos en la [sección 3](/course/chapter7/3), podemos crear lotes con más o menos elementos que el lote de entrada. Esto es útil cuando se hacen operaciones como aumento de datos o filtrado de datos que cambian el número de elementos. En nuestro caso, al tokenizar cada elemento en fragmentos del tamaño de contexto especificado, creamos muchas muestras a partir de cada documento. Solo necesitamos asegurarnos de eliminar las columnas existentes, ya que tienen un tamaño conflictivo. Si quisiéramos mantenerlas, podríamos repetirlas apropiadamente y devolverlas dentro de la llamada `Dataset.map()`:

```py
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['input_ids'],
        num_rows: 16702061
    })
    valid: Dataset({
        features: ['input_ids'],
        num_rows: 93164
    })
})
```

Ahora tenemos 16.7 millones de ejemplos con 128 tokens cada uno, lo que corresponde a aproximadamente 2.1 mil millones de tokens en total. Como referencia, los modelos GPT-3 y Codex de OpenAI se entrenan con 300 y 100 mil millones de tokens, respectivamente, donde los modelos Codex se inicializan desde los checkpoints de GPT-3. Nuestro objetivo en esta sección no es competir con estos modelos, que pueden generar textos largos y coherentes, sino crear una versión reducida que proporcione una función de autocompletado rápido para científicos de datos.

Ahora que tenemos el conjunto de datos listo, ¡configuremos el modelo!

> [!TIP]
> ✏️ **¡Pruébalo!** Deshacerse de todos los fragmentos que son más pequeños que el tamaño del contexto no fue un gran problema aquí porque estamos usando ventanas de contexto pequeñas. A medida que aumentas el tamaño del contexto (o si tienes un corpus de documentos cortos), la fracción de fragmentos que se descartan también crecerá. Una forma más eficiente de preparar los datos es unir todas las muestras tokenizadas en un lote con un token `eos_token_id` entre ellas, y luego realizar el fragmentado en las secuencias concatenadas. Como ejercicio, modifica la función `tokenize()` para hacer uso de ese enfoque. Ten en cuenta que querrás establecer `truncation=False` y eliminar los otros argumentos del tokenizador para obtener la secuencia completa de IDs de tokens.


## Inicializando un nuevo modelo[[initializing-a-new-model]]

Nuestro primer paso es inicializar un modelo GPT-2 desde cero. Usaremos la misma configuración para nuestro modelo que para el modelo GPT-2 pequeño, así que cargamos la configuración preentrenada, nos aseguramos de que el tamaño del tokenizador coincida con el tamaño del vocabulario del modelo y pasamos los IDs de tokens `bos` y `eos` (inicio y fin de secuencia):


**PyTorch:**

```py
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

Con esa configuración, podemos cargar un nuevo modelo. Ten en cuenta que esta es la primera vez que no usamos la función `from_pretrained()`, ya que estamos inicializando un modelo nosotros mismos:

```py
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
```

```python out
GPT-2 size: 124.2M parameters
```

**TensorFlow/Keras:**

```py
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

Con esa configuración, podemos cargar un nuevo modelo. Ten en cuenta que esta es la primera vez que no usamos la función `from_pretrained()`, ya que estamos inicializando un modelo nosotros mismos:

```py
model = TFGPT2LMHeadModel(config)
model(model.dummy_inputs)  # Construye el modelo
model.summary()
```

```python out
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
transformer (TFGPT2MainLayer multiple                  124242432
=================================================================
Total params: 124,242,432
Trainable params: 124,242,432
Non-trainable params: 0
_________________________________________________________________
```


Nuestro modelo tiene 124M de parámetros que tendremos que ajustar. Antes de que podamos comenzar el entrenamiento, necesitamos configurar un data collator que se encargará de crear los lotes. Podemos usar el collator `DataCollatorForLanguageModeling`, que está diseñado específicamente para modelado de lenguaje (como el nombre sugiere sutilmente). Además de apilar y rellenar lotes, también se encarga de crear las etiquetas del modelo de lenguaje -- en el modelado de lenguaje causal las entradas sirven como etiquetas también (solo desplazadas un elemento), y este data collator las crea sobre la marcha durante el entrenamiento para que no necesitemos duplicar los `input_ids`.

Ten en cuenta que `DataCollatorForLanguageModeling` soporta tanto el modelado de lenguaje enmascarado (MLM) como el modelado de lenguaje causal (CLM). Por defecto prepara datos para MLM, pero podemos cambiar a CLM estableciendo el argumento `mlm=False`:


**PyTorch:**

```py
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

**TensorFlow/Keras:**

```py
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")
```


Veamos un ejemplo:

```py
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
```


**PyTorch:**

```python out
input_ids shape: torch.Size([5, 128])
attention_mask shape: torch.Size([5, 128])
labels shape: torch.Size([5, 128])
```

**TensorFlow/Keras:**

```python out
input_ids shape: (5, 128)
attention_mask shape: (5, 128)
labels shape: (5, 128)
```


Podemos ver que los ejemplos han sido apilados y todos los tensores tienen la misma forma.

{#if fw === 'tf'}

Ahora podemos usar el método `prepare_tf_dataset()` para convertir nuestros conjuntos de datos a conjuntos de datos de TensorFlow con el data collator que creamos arriba:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["valid"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)
```

{/if}

> [!WARNING]
> ⚠️ El desplazamiento de las entradas y etiquetas para alinearlas ocurre dentro del modelo, así que el data collator simplemente copia las entradas para crear las etiquetas.


Ahora tenemos todo en su lugar para entrenar nuestro modelo -- ¡no fue tanto trabajo después de todo! Antes de comenzar el entrenamiento, debemos iniciar sesión en Hugging Face. Si estás trabajando en un notebook, puedes hacerlo con la siguiente función utilitaria:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Esto mostrará un widget donde puedes ingresar tus credenciales de inicio de sesión de Hugging Face.

Si no estás trabajando en un notebook, simplemente escribe la siguiente línea en tu terminal:

```bash
huggingface-cli login
```


**PyTorch:**

Todo lo que queda por hacer es configurar los argumentos de entrenamiento e iniciar el `Trainer`. Usaremos un programa de tasa de aprendizaje de coseno con algo de calentamiento y un tamaño de lote efectivo de 256 (`per_device_train_batch_size` * `gradient_accumulation_steps`). La acumulación de gradientes se usa cuando un solo lote no cabe en memoria, y construye incrementalmente el gradiente a través de varios pases hacia adelante/atrás. Veremos esto en acción cuando creemos el bucle de entrenamiento con 🤗 Accelerate.

```py
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
```

Ahora simplemente podemos iniciar el `Trainer` y esperar a que termine el entrenamiento. Dependiendo de si lo ejecutas en el conjunto de entrenamiento completo o en un subconjunto, esto tomará 20 o 2 horas, respectivamente, ¡así que toma algunos cafés y un buen libro para leer!

```py
trainer.train()
```

Después de que el entrenamiento se complete, podemos subir el modelo y el tokenizador al Hub:

```py
trainer.push_to_hub()
```

**TensorFlow/Keras:**

Todo lo que queda por hacer es configurar los hiperparámetros de entrenamiento y llamar a `compile()` y `fit()`. Usaremos un programa de tasa de aprendizaje con algo de calentamiento para mejorar la estabilidad del entrenamiento:

```py
from transformers import create_optimizer
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Entrenar en precisión mixta float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Ahora simplemente podemos llamar a `model.fit()` y esperar a que termine el entrenamiento. Dependiendo de si lo ejecutas en el conjunto de entrenamiento completo o en un subconjunto, esto tomará 20 o 2 horas, respectivamente, ¡así que toma algunos cafés y un buen libro para leer! Después de que el entrenamiento se complete, podemos subir el modelo y el tokenizador al Hub:

```py
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="codeparrot-ds", tokenizer=tokenizer)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
```


> [!TIP]
> ✏️ **¡Pruébalo!** Solo nos tomó alrededor de 30 líneas de código además de los `TrainingArguments` para pasar de textos en bruto a entrenar GPT-2. ¡Pruébalo con tu propio conjunto de datos y ve si puedes obtener buenos resultados!

> [!TIP]
> 
**PyTorch:**

>
> 💡 Si tienes acceso a una máquina con múltiples GPUs, intenta ejecutar el código allí. El `Trainer` gestiona automáticamente múltiples máquinas, y esto puede acelerar el entrenamiento tremendamente.
>
>

**TensorFlow/Keras:**

>
> 💡 Si tienes acceso a una máquina con múltiples GPUs, puedes intentar usar un contexto `MirroredStrategy` para acelerar sustancialmente el entrenamiento. Necesitarás crear un objeto `tf.distribute.MirroredStrategy`, y asegurarte de que cualquier método `to_tf_dataset()` o `prepare_tf_dataset()` así como la creación del modelo y la llamada a `fit()` se ejecuten dentro de su contexto `scope()`. Puedes ver la documentación sobre esto [aquí](https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_keras_modelfit).
>
>


## Generación de código con un pipeline[[code-generation-with-a-pipeline]]

Ahora es el momento de la verdad: ¡veamos qué tan bien funciona realmente el modelo entrenado! Podemos ver en los logs que la pérdida disminuyó de manera constante, pero para poner el modelo a prueba veamos qué tan bien funciona en algunos prompts. Para hacer eso, envolveremos el modelo en un `pipeline` de generación de texto, y lo pondremos en la GPU para generaciones rápidas si hay una disponible:


**PyTorch:**

```py
import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)
```

**TensorFlow/Keras:**

```py
from transformers import pipeline

course_model = TFGPT2LMHeadModel.from_pretrained("huggingface-course/codeparrot-ds")
course_tokenizer = AutoTokenizer.from_pretrained("huggingface-course/codeparrot-ds")
pipe = pipeline(
    "text-generation", model=course_model, tokenizer=course_tokenizer, device=0
)
```


Comencemos con la tarea simple de crear un gráfico de dispersión:

```py
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
plt.scatter(x, y)

# create scatter
```

El resultado parece correcto. ¿También funciona para una operación de `pandas`? Veamos si podemos crear un `DataFrame` a partir de dos arrays:

```py
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
df = pd.DataFrame({'x': x, 'y': y})
df.insert(0,'x', x)
for
```

Bien, esa es la respuesta correcta -- aunque luego inserta la columna `x` de nuevo. Como el número de tokens generados es limitado, el siguiente bucle `for` se corta. Veamos si podemos hacer algo un poco más complejo y hacer que el modelo nos ayude a usar la operación `groupby`:

```py
txt = """\
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
profession = df.groupby(['profession']).mean()

# compute the
```

Nada mal; esa es la forma correcta de hacerlo. Finalmente, veamos si también podemos usarlo para `scikit-learn` y configurar un modelo de Random Forest:

```py
txt = """
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
rf = RandomForestRegressor(n_estimators=300, random_state=random_state, max_depth=3)
rf.fit(X, y)
rf
```

{#if fw === 'tf'}

Mirando estos pocos ejemplos, parece que el modelo ha aprendido algo de la sintaxis del stack de ciencia de datos de Python. Por supuesto, necesitaríamos evaluar el modelo más a fondo antes de desplegarlo en el mundo real, pero esto sigue siendo un prototipo impresionante.

{:else}

Mirando estos pocos ejemplos, parece que el modelo ha aprendido algo de la sintaxis del stack de ciencia de datos de Python (por supuesto, necesitaríamos evaluarlo más a fondo antes de desplegar el modelo en el mundo real). Sin embargo, a veces se requiere más personalización del entrenamiento del modelo para lograr el rendimiento necesario para un caso de uso dado. Por ejemplo, ¿qué pasa si quisiéramos actualizar dinámicamente el tamaño del lote o tener un bucle de entrenamiento condicional que salte ejemplos malos sobre la marcha? Una opción sería crear una subclase del `Trainer` y agregar los cambios necesarios, pero a veces es más simple escribir el bucle de entrenamiento desde cero. Ahí es donde entra 🤗 Accelerate.

{/if}


**PyTorch:**

## Entrenamiento con 🤗 Accelerate[[training-with-accelerate]]

Hemos visto cómo entrenar un modelo con el `Trainer`, que puede permitir cierta personalización. Sin embargo, a veces queremos control total sobre el bucle de entrenamiento, o queremos hacer algunos cambios exóticos. En este caso, 🤗 Accelerate es una gran opción, y en esta sección repasaremos los pasos para usarlo para entrenar nuestro modelo. Para hacer las cosas más interesantes, también agregaremos un giro al bucle de entrenamiento.


**Video:** [Ver en YouTube](https://youtu.be/Hm8_PgVTFuc)


Como estamos principalmente interesados en autocompletado sensato para las bibliotecas de ciencia de datos, tiene sentido dar más peso a las muestras de entrenamiento que hacen más uso de estas bibliotecas. Podemos identificar fácilmente estos ejemplos a través del uso de palabras clave como `plt`, `pd`, `sk`, `fit` y `predict`, que son los nombres de importación más frecuentes para `matplotlib.pyplot`, `pandas` y `sklearn`, así como el patrón fit/predict de este último. Si cada uno de estos está representado como un solo token, podemos verificar fácilmente si ocurren en la secuencia de entrada. Los tokens pueden tener un prefijo de espacio en blanco, así que también verificaremos esas versiones en el vocabulario del tokenizador. Para verificar que funciona, agregaremos un token de prueba que debería dividirse en múltiples tokens:

```py
keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")
```

```python out
'Keyword has not single token: testtest'
```

Genial, ¡eso parece funcionar bien! Ahora podemos escribir una función de pérdida personalizada que tome la secuencia de entrada, los logits y los tokens clave que acabamos de seleccionar como entradas. Primero necesitamos alinear los logits y las entradas: la secuencia de entrada desplazada una posición a la derecha forma las etiquetas, ya que el siguiente token es la etiqueta para el token actual. Podemos lograr esto comenzando las etiquetas desde el segundo token de la secuencia de entrada, ya que el modelo no hace una predicción para el primer token de todos modos. Luego cortamos el último logit, ya que no tenemos una etiqueta para el token que sigue a la secuencia de entrada completa. Con eso podemos calcular la pérdida por muestra y contar las ocurrencias de todas las palabras clave en cada muestra. Finalmente, calculamos el promedio ponderado sobre todas las muestras usando las ocurrencias como pesos. Como no queremos descartar todas las muestras que no tienen palabras clave, añadimos 1 a los pesos:

```py
from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Desplazar para que los tokens < n predigan n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calcular la pérdida por token
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Redimensionar y promediar la pérdida por muestra
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calcular y escalar la ponderación
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calcular el promedio ponderado
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
```

Antes de que podamos comenzar a entrenar con esta increíble nueva función de pérdida, necesitamos preparar algunas cosas:

- Necesitamos dataloaders para cargar los datos en lotes.
- Necesitamos configurar los parámetros de decaimiento de pesos.
- De vez en cuando queremos evaluar, así que tiene sentido envolver el código de evaluación en una función.

Comencemos con los dataloaders. Solo necesitamos establecer el formato del conjunto de datos a `"torch"`, y luego podemos pasarlo a un `DataLoader` de PyTorch con el tamaño de lote apropiado:

```py
from torch.utils.data.dataloader import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)
```

A continuación, agrupamos los parámetros para que el optimizador sepa cuáles tendrán un decaimiento de peso adicional. Usualmente, todos los términos de bias y pesos de LayerNorm están exentos de esto; así es como podemos hacer esto:

```py
weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
```

Como queremos evaluar el modelo regularmente en el conjunto de validación durante el entrenamiento, escribamos también una función para eso. Simplemente recorre el dataloader de evaluación y recopila todas las pérdidas a través de los procesos:

```py
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
```

Con la función `evaluate()` podemos reportar la pérdida y la [perplejidad](/course/chapter7/3) a intervalos regulares. A continuación, redefinimos nuestro modelo para asegurarnos de que entrenamos desde cero nuevamente:

```py
model = GPT2LMHeadModel(config)
```

Luego podemos definir nuestro optimizador, usando la función de antes para separar los parámetros para el decaimiento de pesos:

```py
from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)
```

Ahora preparemos el modelo, optimizador y dataloaders para que podamos comenzar a entrenar:

```py
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> 🚨 Si estás entrenando en una TPU, necesitarás mover todo el código comenzando desde la celda de arriba a una función de entrenamiento dedicada. Consulta el [Capítulo 3](/course/chapter3) para más detalles.

Ahora que hemos enviado nuestro `train_dataloader` a `accelerator.prepare()`, podemos usar su longitud para calcular el número de pasos de entrenamiento. Recuerda que siempre debemos hacer esto después de preparar el dataloader, ya que ese método cambiará su longitud. Usamos un programa lineal clásico desde la tasa de aprendizaje hasta 0:

```py
from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)
```

Por último, para subir nuestro modelo al Hub, necesitaremos crear un objeto `Repository` en una carpeta de trabajo. Primero inicia sesión en el Hugging Face Hub, si no has iniciado sesión ya. Determinaremos el nombre del repositorio a partir del ID del modelo que queremos darle a nuestro modelo (siéntete libre de reemplazar el `repo_name` con tu propia elección; solo necesita contener tu nombre de usuario, que es lo que hace la función `get_full_repo_name()`):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "codeparrot-ds-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'sgugger/codeparrot-ds-accelerate'
```

Luego podemos clonar ese repositorio en una carpeta local. Si ya existe, esta carpeta local debería ser un clon existente del repositorio con el que estamos trabajando:

```py
output_dir = "codeparrot-ds-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

Ahora podemos subir cualquier cosa que guardemos en `output_dir` llamando al método `repo.push_to_hub()`. Esto nos ayudará a subir los modelos intermedios al final de cada época.

Antes de entrenar, ejecutemos una prueba rápida para ver si la función de evaluación funciona correctamente:

```py
evaluate()
```

```python out
(10.934126853942871, 56057.14453125)
```

Esos son valores muy altos para la pérdida y la perplejidad, pero eso no es sorprendente ya que aún no hemos entrenado el modelo. Con eso, tenemos todo preparado para escribir la parte central del script de entrenamiento: el bucle de entrenamiento. En el bucle de entrenamiento iteramos sobre el dataloader y pasamos los lotes al modelo. Con los logits, podemos entonces evaluar nuestra función de pérdida personalizada. Escalamos la pérdida por el número de pasos de acumulación de gradientes para no crear pérdidas más grandes al agregar más pasos. Antes de optimizar, también recortamos los gradientes para una mejor convergencia. Finalmente, cada ciertos pasos evaluamos el modelo en el conjunto de evaluación con nuestra nueva función `evaluate()`:

```py
from tqdm.notebook import tqdm

gradient_accumulation_steps = 8
eval_steps = 5_000

model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress step {step}", blocking=False
                )
```

¡Y eso es todo! Ahora tienes tu propio bucle de entrenamiento personalizado para modelos de lenguaje causal como GPT-2 que puedes personalizar aún más según tus necesidades.

> [!TIP]
> ✏️ **¡Pruébalo!** Crea tu propia función de pérdida personalizada adaptada a tu caso de uso, o añade otro paso personalizado en el bucle de entrenamiento.

> [!TIP]
> ✏️ **¡Pruébalo!** Cuando ejecutas experimentos de entrenamiento largos, es una buena idea registrar métricas importantes usando herramientas como TensorBoard o Weights & Biases. Añade un registro adecuado al bucle de entrenamiento para que siempre puedas verificar cómo va el entrenamiento.



---



# Respuesta a preguntas[[question-answering]]


Es momento de explorar la respuesta a preguntas. Esta tarea viene en muchas variantes, pero en la que nos enfocaremos en esta seccion se llama respuesta a preguntas *extractiva*. Esto implica plantear preguntas sobre un documento e identificar las respuestas como _fragmentos de texto_ dentro del propio documento.


**Video:** [Ver en YouTube](https://youtu.be/ajPx5LwJD-I)


Vamos a ajustar finamente un modelo BERT en el [conjunto de datos SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), que consiste en preguntas planteadas por trabajadores de crowdsourcing sobre un conjunto de articulos de Wikipedia. Esto nos dara un modelo capaz de calcular predicciones como esta:

<iframe src="https://course-demos-bert-finetuned-squad.hf.space" frameBorder="0" height="450" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

Esto en realidad muestra el modelo que fue entrenado y subido al Hub usando el codigo mostrado en esta seccion. Puedes encontrarlo y verificar las predicciones [aqui](https://huggingface.co/huggingface-course/bert-finetuned-squad?context=%F0%9F%A4%97+Transformers+is+backed+by+the+three+most+popular+deep+learning+libraries+%E2%80%94+Jax%2C+PyTorch+and+TensorFlow+%E2%80%94+with+a+seamless+integration+between+them.+It%27s+straightforward+to+train+your+models+with+one+before+loading+them+for+inference+with+the+other.&question=Which+deep+learning+libraries+back+%F0%9F%A4%97+Transformers%3F).

> [!TIP]
> Los modelos solo de codificador como BERT tienden a ser excelentes para extraer respuestas a preguntas factuales como "Quien invento la arquitectura Transformer?" pero les va mal cuando se les dan preguntas abiertas como "Por que el cielo es azul?" En estos casos mas desafiantes, los modelos codificador-decodificador como T5 y BART se usan tipicamente para sintetizar la informacion de una manera bastante similar a la [resumenes de texto](/course/chapter7/5). Si te interesa este tipo de respuesta a preguntas *generativa*, te recomendamos revisar nuestra [demo](https://yjernite.github.io/lfqa.html) basada en el [conjunto de datos ELI5](https://huggingface.co/datasets/eli5).

## Preparando los datos[[preparing-the-data]]

El conjunto de datos que mas se usa como referencia academica para la respuesta a preguntas extractiva es [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), asi que ese es el que usaremos aqui. Tambien existe una referencia mas dificil llamada [SQuAD v2](https://huggingface.co/datasets/squad_v2), que incluye preguntas que no tienen respuesta. Siempre que tu propio conjunto de datos contenga una columna para contextos, una columna para preguntas y una columna para respuestas, deberias poder adaptar los pasos siguientes.

### El conjunto de datos SQuAD[[the-squad-dataset]]

Como es usual, podemos descargar y almacenar en cache el conjunto de datos en un solo paso gracias a `load_dataset()`:

```py
from datasets import load_dataset

raw_datasets = load_dataset("squad")
```

Luego podemos echar un vistazo a este objeto para aprender mas sobre el conjunto de datos SQuAD:

```py
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
```

Parece que tenemos todo lo que necesitamos con los campos `context`, `question` y `answers`, asi que imprimamos esos para el primer elemento de nuestro conjunto de entrenamiento:

```py
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])
```

```python out
Context: 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
Question: 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
Answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
```

Los campos `context` y `question` son muy sencillos de usar. El campo `answers` es un poco mas complicado ya que contiene un diccionario con dos campos que son ambos listas. Este es el formato que sera esperado por la metrica `squad` durante la evaluacion; si estas usando tus propios datos, no necesariamente tienes que preocuparte por poner las respuestas en el mismo formato. El campo `text` es bastante obvio, y el campo `answer_start` contiene el indice del caracter inicial de cada respuesta en el contexto.

Durante el entrenamiento, solo hay una respuesta posible. Podemos verificar esto usando el metodo `Dataset.filter()`:

```py
raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
```

```python out
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers'],
    num_rows: 0
})
```

Sin embargo, para la evaluacion hay varias respuestas posibles para cada muestra, que pueden ser iguales o diferentes:

```py
print(raw_datasets["validation"][0]["answers"])
print(raw_datasets["validation"][2]["answers"])
```

```python out
{'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'], 'answer_start': [177, 177, 177]}
{'text': ['Santa Clara, California', "Levi's Stadium", "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."], 'answer_start': [403, 355, 355]}
```

No profundizaremos en el script de evaluacion ya que todo sera envuelto por una metrica de Datasets para nosotros, pero la version corta es que algunas de las preguntas tienen varias respuestas posibles, y este script comparara una respuesta predicha con todas las respuestas aceptables y tomara la mejor puntuacion. Si miramos la muestra en el indice 2, por ejemplo:

```py
print(raw_datasets["validation"][2]["context"])
print(raw_datasets["validation"][2]["question"])
```

```python out
'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
'Where did Super Bowl 50 take place?'
```

podemos ver que la respuesta puede ser efectivamente una de las tres posibilidades que vimos antes.

### Procesando los datos de entrenamiento[[processing-the-training-data]]


**Video:** [Ver en YouTube](https://youtu.be/qgaM0weJHpA)


Comencemos con el preprocesamiento de los datos de entrenamiento. La parte dificil sera generar etiquetas para la respuesta de la pregunta, que seran las posiciones de inicio y fin de los tokens correspondientes a la respuesta dentro del contexto.

Pero no nos adelantemos. Primero, necesitamos convertir el texto en la entrada en IDs que el modelo pueda entender, usando un tokenizador:

```py
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Como se menciono anteriormente, vamos a ajustar finamente un modelo BERT, pero puedes usar cualquier otro tipo de modelo siempre que tenga un tokenizador rapido implementado. Puedes ver todas las arquitecturas que vienen con una version rapida en [esta gran tabla](https://huggingface.co/transformers/#supported-frameworks), y para verificar que el objeto `tokenizer` que estas usando esta respaldado por Tokenizers puedes mirar su atributo `is_fast`:

```py
tokenizer.is_fast
```

```python out
True
```

Podemos pasar a nuestro tokenizador la pregunta y el contexto juntos, y este insertara correctamente los tokens especiales para formar una oracion como esta:

```
[CLS] question [SEP] context [SEP]
```

Verifiquemos:

```py
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])
```

```python out
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Architecturally, '
'the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin '
'Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms '
'upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred '
'Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a '
'replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette '
'Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 statues '
'and the Gold Dome ), is a simple, modern stone statue of Mary. [SEP]'
```

Las etiquetas seran entonces el indice de los tokens que inician y terminan la respuesta, y el modelo tendra la tarea de predecir un logit de inicio y fin por cada token en la entrada, con las etiquetas teoricas siendo las siguientes:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/qa_labels.svg" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/qa_labels-dark.svg" alt="One-hot encoded labels for question answering."/>
</div>

En este caso el contexto no es muy largo, pero algunos de los ejemplos en el conjunto de datos tienen contextos muy largos que excederan la longitud maxima que establecimos (que es 384 en este caso). Como vimos en el [Capitulo 6](/course/chapter6/4) cuando exploramos los internos del pipeline `question-answering`, manejaremos los contextos largos creando varias caracteristicas de entrenamiento a partir de una muestra de nuestro conjunto de datos, con una ventana deslizante entre ellas.

Para ver como funciona esto usando el ejemplo actual, podemos limitar la longitud a 100 y usar una ventana deslizante de 50 tokens. Como recordatorio, usamos:

- `max_length` para establecer la longitud maxima (aqui 100)
- `truncation="only_second"` para truncar el contexto (que esta en la segunda posicion) cuando la pregunta con su contexto es muy larga
- `stride` para establecer el numero de tokens superpuestos entre dos fragmentos sucesivos (aqui 50)
- `return_overflowing_tokens=True` para que el tokenizador sepa que queremos los tokens desbordantes

```py
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

```python out
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basi [SEP]'
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin [SEP]'
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 [SEP]'
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP]. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 statues and the Gold Dome ), is a simple, modern stone statue of Mary. [SEP]'
```

Como podemos ver, nuestro ejemplo se ha dividido en cuatro entradas, cada una de ellas conteniendo la pregunta y alguna parte del contexto. Ten en cuenta que la respuesta a la pregunta ("Bernadette Soubirous") solo aparece en la tercera y ultima entrada, asi que al manejar contextos largos de esta manera crearemos algunos ejemplos de entrenamiento donde la respuesta no esta incluida en el contexto. Para esos ejemplos, las etiquetas seran `start_position = end_position = 0` (asi que predecimos el token `[CLS]`). Tambien estableceremos esas etiquetas en el desafortunado caso de que la respuesta haya sido truncada de modo que solo tengamos el inicio (o el fin) de ella. Para los ejemplos donde la respuesta esta completamente en el contexto, las etiquetas seran el indice del token donde comienza la respuesta y el indice del token donde termina la respuesta.

El conjunto de datos nos proporciona el caracter de inicio de la respuesta en el contexto, y sumando la longitud de la respuesta, podemos encontrar el caracter final en el contexto. Para mapear estos a indices de tokens, necesitaremos usar los mapeos de offset que estudiamos en el [Capitulo 6](/course/chapter6/4). Podemos hacer que nuestro tokenizador los devuelva pasando `return_offsets_mapping=True`:

```py
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
inputs.keys()
```

```python out
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
```

Como podemos ver, obtenemos los IDs de entrada usuales, los IDs de tipo de token y la mascara de atencion, asi como el mapeo de offset que solicitamos y una clave extra, `overflow_to_sample_mapping`. El valor correspondiente nos sera util cuando tokenicemos varios textos al mismo tiempo (lo cual debemos hacer para beneficiarnos del hecho de que nuestro tokenizador esta respaldado por Rust). Como una muestra puede dar varias caracteristicas, mapea cada caracteristica al ejemplo del que se origino. Porque aqui solo tokenizamos un ejemplo, obtenemos una lista de `0`s:

```py
inputs["overflow_to_sample_mapping"]
```

```python out
[0, 0, 0, 0]
```

Pero si tokenizamos mas ejemplos, esto se volvera mas util:

```py
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
```

```python out
'The 4 examples gave 19 features.'
'Here is where each comes from: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3].'
```

Como podemos ver, los primeros tres ejemplos (en los indices 2, 3 y 4 del conjunto de entrenamiento) dieron cada uno cuatro caracteristicas y el ultimo ejemplo (en el indice 5 del conjunto de entrenamiento) dio 7 caracteristicas.

Esta informacion sera util para mapear cada caracteristica que obtengamos a su etiqueta correspondiente. Como se menciono antes, esas etiquetas son:

- `(0, 0)` si la respuesta no esta en el fragmento correspondiente del contexto
- `(start_position, end_position)` si la respuesta esta en el fragmento correspondiente del contexto, con `start_position` siendo el indice del token (en los IDs de entrada) al inicio de la respuesta y `end_position` siendo el indice del token (en los IDs de entrada) donde termina la respuesta

Para determinar cual de estos es el caso y, si es relevante, las posiciones de los tokens, primero encontramos los indices que inician y terminan el contexto en los IDs de entrada. Podriamos usar los IDs de tipo de token para hacer esto, pero como esos no necesariamente existen para todos los modelos (DistilBERT no los requiere, por ejemplo), en su lugar usaremos el metodo `sequence_ids()` del `BatchEncoding` que nuestro tokenizador devuelve.

Una vez que tenemos esos indices de token, miramos los offsets correspondientes, que son tuplas de dos enteros que representan el fragmento de caracteres dentro del contexto original. Asi podemos detectar si el fragmento del contexto en esta caracteristica comienza despues de la respuesta o termina antes de que la respuesta comience (en cuyo caso la etiqueta es `(0, 0)`). Si ese no es el caso, iteramos para encontrar el primer y ultimo token de la respuesta:

```py
answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Encontrar el inicio y fin del contexto
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # Si la respuesta no esta completamente dentro del contexto, la etiqueta es (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # De lo contrario son las posiciones de inicio y fin del token
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

start_positions, end_positions
```

```python out
([83, 51, 19, 0, 0, 64, 27, 0, 34, 0, 0, 0, 67, 34, 0, 0, 0, 0, 0],
 [85, 53, 21, 0, 0, 70, 33, 0, 40, 0, 0, 0, 68, 35, 0, 0, 0, 0, 0])
```

Echemos un vistazo a algunos resultados para verificar que nuestro enfoque es correcto. Para la primera caracteristica encontramos `(83, 85)` como etiquetas, asi que comparemos la respuesta teorica con el fragmento decodificado de tokens del 83 al 85 (inclusive):

```py
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")
```

```python out
'Theoretical answer: the Main Building, labels give: the Main Building'
```

Entonces coincide! Ahora verifiquemos el indice 4, donde establecimos las etiquetas a `(0, 0)`, lo que significa que la respuesta no esta en el fragmento de contexto de esa caracteristica:

```py
idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")
```

```python out
'Theoretical answer: a Marian place of prayer and reflection, decoded example: [CLS] What is the Grotto at Notre Dame? [SEP] Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grot [SEP]'
```

Efectivamente, no vemos la respuesta dentro del contexto.

> [!TIP]
> Tu turno! Cuando uses la arquitectura XLNet, el padding se aplica a la izquierda y la pregunta y el contexto se intercambian. Adapta todo el codigo que acabamos de ver a la arquitectura XLNet (y agrega `padding=True`). Ten en cuenta que el token `[CLS]` puede no estar en la posicion 0 con el padding aplicado.

Ahora que hemos visto paso a paso como preprocesar nuestros datos de entrenamiento, podemos agruparlo en una funcion que aplicaremos a todo el conjunto de datos de entrenamiento. Rellenaremos cada caracteristica a la longitud maxima que establecimos, ya que la mayoria de los contextos seran largos (y las muestras correspondientes se dividiran en varias caracteristicas), por lo que no hay un beneficio real en aplicar padding dinamico aqui:

```py
max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Encontrar el inicio y fin del contexto
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Si la respuesta no esta completamente dentro del contexto, la etiqueta es (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # De lo contrario son las posiciones de inicio y fin del token
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
```

Ten en cuenta que definimos dos constantes para determinar la longitud maxima usada asi como la longitud de la ventana deslizante, y que agregamos una pequena limpieza antes de tokenizar: algunas de las preguntas en el conjunto de datos SQuAD tienen espacios extra al principio y al final que no agregan nada (y ocupan espacio cuando se tokenizan si usas un modelo como RoBERTa), por lo que eliminamos esos espacios extra.

Para aplicar esta funcion a todo el conjunto de entrenamiento, usamos el metodo `Dataset.map()` con la bandera `batched=True`. Es necesario aqui ya que estamos cambiando la longitud del conjunto de datos (ya que un ejemplo puede dar varias caracteristicas de entrenamiento):

```py
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
len(raw_datasets["train"]), len(train_dataset)
```

```python out
(87599, 88729)
```

Como podemos ver, el preprocesamiento agrego aproximadamente 1,000 caracteristicas. Nuestro conjunto de entrenamiento ahora esta listo para usarse -- profundicemos en el preprocesamiento del conjunto de validacion!

### Procesando los datos de validacion[[processing-the-validation-data]]

Preprocesar los datos de validacion sera un poco mas facil ya que no necesitamos generar etiquetas (a menos que queramos calcular una perdida de validacion, pero ese numero realmente no nos ayudara a entender que tan bueno es el modelo). La verdadera alegria sera interpretar las predicciones del modelo en fragmentos del contexto original. Para esto, solo necesitaremos almacenar tanto los mapeos de offset como alguna forma de hacer coincidir cada caracteristica creada con el ejemplo original del que proviene. Como hay una columna de ID en el conjunto de datos original, usaremos ese ID.

Lo unico que agregaremos aqui es una pequena limpieza de los mapeos de offset. Contendran offsets para la pregunta y el contexto, pero una vez que estemos en la etapa de post-procesamiento no tendremos ninguna forma de saber que parte de los IDs de entrada correspondia al contexto y cual parte era la pregunta (el metodo `sequence_ids()` que usamos solo esta disponible para la salida del tokenizador). Por lo tanto, estableceremos los offsets correspondientes a la pregunta en `None`:

```py
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs
```

Podemos aplicar esta funcion en todo el conjunto de datos de validacion como antes:

```py
validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
len(raw_datasets["validation"]), len(validation_dataset)
```

```python out
(10570, 10822)
```

En este caso solo agregamos un par de cientos de muestras, por lo que parece que los contextos en el conjunto de datos de validacion son un poco mas cortos.

Ahora que hemos preprocesado todos los datos, podemos pasar al entrenamiento.


**PyTorch:**

## Ajuste fino del modelo con la API Trainer[[fine-tuning-the-model-with-the-trainer-api]]

El codigo de entrenamiento para este ejemplo se parecera mucho al codigo de las secciones anteriores -- lo mas dificil sera escribir la funcion `compute_metrics()`. Como rellenamos todas las muestras a la longitud maxima que establecimos, no hay un data collator que definir, por lo que este calculo de metricas es realmente lo unico de lo que debemos preocuparnos. La parte dificil sera post-procesar las predicciones del modelo en fragmentos de texto en los ejemplos originales; una vez que hayamos hecho eso, la metrica de la biblioteca Datasets hara la mayor parte del trabajo por nosotros.

**TensorFlow/Keras:**

## Ajuste fino del modelo con Keras[[fine-tuning-the-model-with-keras]]

El codigo de entrenamiento para este ejemplo se parecera mucho al codigo de las secciones anteriores, pero calcular las metricas sera un desafio unico. Como rellenamos todas las muestras a la longitud maxima que establecimos, no hay un data collator que definir, por lo que este calculo de metricas es realmente lo unico de lo que debemos preocuparnos. La parte dificil sera post-procesar las predicciones del modelo en fragmentos de texto en los ejemplos originales; una vez que hayamos hecho eso, la metrica de la biblioteca Datasets hara la mayor parte del trabajo por nosotros.


### Post-procesamiento[[post-processing]]


**PyTorch:**

**Video:** [Ver en YouTube](https://youtu.be/BNy08iIWVJM)

**TensorFlow/Keras:**

**Video:** [Ver en YouTube](https://youtu.be/VN67ZpN33Ss)


El modelo producira logits para las posiciones de inicio y fin de la respuesta en los IDs de entrada, como vimos durante nuestra exploracion del [pipeline `question-answering`](/course/chapter6/3b). El paso de post-procesamiento sera similar a lo que hicimos alli, asi que aqui hay un rapido recordatorio de las acciones que tomamos:

- Enmascaramos los logits de inicio y fin correspondientes a tokens fuera del contexto.
- Luego convertimos los logits de inicio y fin en probabilidades usando un softmax.
- Atribuimos una puntuacion a cada par `(start_token, end_token)` tomando el producto de las dos probabilidades correspondientes.
- Buscamos el par con la puntuacion maxima que produjo una respuesta valida (por ejemplo, un `start_token` menor que `end_token`).

Aqui cambiaremos este proceso ligeramente porque no necesitamos calcular puntuaciones reales (solo la respuesta predicha). Esto significa que podemos omitir el paso del softmax. Para ir mas rapido, tampoco puntuaremos todos los posibles pares `(start_token, end_token)`, sino solo los que corresponden a los `n_best` logits mas altos (con `n_best=20`). Como omitiremos el softmax, esas puntuaciones seran puntuaciones de logits, y se obtendran tomando la suma de los logits de inicio y fin (en lugar del producto, debido a la regla \\(\log(ab) = \log(a) + \log(b)\\)).

Para demostrar todo esto, necesitaremos algun tipo de predicciones. Como aun no hemos entrenado nuestro modelo, vamos a usar el modelo predeterminado del pipeline de QA para generar algunas predicciones en una pequena parte del conjunto de validacion. Podemos usar la misma funcion de procesamiento que antes; porque depende de la constante global `tokenizer`, solo tenemos que cambiar ese objeto al tokenizador del modelo que queremos usar temporalmente:

```python
small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
```

Ahora que el preprocesamiento esta hecho, cambiamos el tokenizador de vuelta al que escogimos originalmente:

```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Luego removemos las columnas de nuestro `eval_set` que no son esperadas por el modelo, construimos un lote con todo ese pequeno conjunto de validacion, y lo pasamos a traves del modelo. Si hay una GPU disponible, la usamos para ir mas rapido:


**PyTorch:**

```python
import torch
from transformers import AutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
    device
)

with torch.no_grad():
    outputs = trained_model(**batch)
```

Como el `Trainer` nos dara predicciones como arrays de NumPy, tomamos los logits de inicio y fin y los convertimos a ese formato:

```python
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()
```

**TensorFlow/Keras:**

```python
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("numpy")

batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}
trained_model = TFAutoModelForQuestionAnswering.from_pretrained(trained_checkpoint)

outputs = trained_model(**batch)
```

Para facilitar la experimentacion, convirtamos estas salidas a arrays de NumPy:

```python
start_logits = outputs.start_logits.numpy()
end_logits = outputs.end_logits.numpy()
```


Ahora, necesitamos encontrar la respuesta predicha para cada ejemplo en nuestro `small_eval_set`. Un ejemplo puede haber sido dividido en varias caracteristicas en `eval_set`, por lo que el primer paso es mapear cada ejemplo en `small_eval_set` a las caracteristicas correspondientes en `eval_set`:

```python
import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)
```

Con esto en mano, realmente podemos ponernos a trabajar iterando a traves de todos los ejemplos y, para cada ejemplo, a traves de todas las caracteristicas asociadas. Como dijimos antes, miraremos las puntuaciones de logits para los `n_best` logits de inicio y logits de fin, excluyendo posiciones que dan:

- Una respuesta que no estaria dentro del contexto
- Una respuesta con longitud negativa
- Una respuesta que es muy larga (limitamos las posibilidades a `max_answer_length=30`)

Una vez que tenemos todas las posibles respuestas puntuadas para un ejemplo, simplemente escogemos la que tiene la mejor puntuacion de logits:

```python
import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Omitir respuestas que no estan completamente en el contexto
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Omitir respuestas con una longitud que es < 0 o > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
```

El formato final de las respuestas predichas es el que sera esperado por la metrica que usaremos. Como es usual, podemos cargarla con la ayuda de la biblioteca Evaluate:

```python
import evaluate

metric = evaluate.load("squad")
```

Esta metrica espera las respuestas predichas en el formato que vimos arriba (una lista de diccionarios con una clave para el ID del ejemplo y una clave para el texto predicho) y las respuestas teoricas en el formato de abajo (una lista de diccionarios con una clave para el ID del ejemplo y una clave para las posibles respuestas):

```python
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
```

Ahora podemos verificar que obtenemos resultados razonables mirando el primer elemento de ambas listas:

```python
print(predicted_answers[0])
print(theoretical_answers[0])
```

```python out
{'id': '56be4db0acb8001400a502ec', 'prediction_text': 'Denver Broncos'}
{'id': '56be4db0acb8001400a502ec', 'answers': {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'], 'answer_start': [177, 177, 177]}}
```

Nada mal! Ahora echemos un vistazo a la puntuacion que nos da la metrica:

```python
metric.compute(predictions=predicted_answers, references=theoretical_answers)
```

```python out
{'exact_match': 83.0, 'f1': 88.25}
```

De nuevo, eso es bastante bueno considerando que segun [su articulo](https://arxiv.org/abs/1910.01108v2) DistilBERT ajustado finamente en SQuAD obtiene 79.1 y 86.9 para esas puntuaciones en todo el conjunto de datos.


**PyTorch:**

Ahora pongamos todo lo que acabamos de hacer en una funcion `compute_metrics()` que usaremos en el `Trainer`. Normalmente, esa funcion `compute_metrics()` solo recibe una tupla `eval_preds` con logits y etiquetas. Aqui necesitaremos un poco mas, ya que tenemos que buscar en el conjunto de datos de caracteristicas los offsets y en el conjunto de datos de ejemplos los contextos originales, por lo que no podremos usar esta funcion para obtener resultados de evaluacion regulares durante el entrenamiento. Solo la usaremos al final del entrenamiento para verificar los resultados.

La funcion `compute_metrics()` agrupa los mismos pasos que antes; solo agregamos una pequena verificacion en caso de que no encontremos ninguna respuesta valida (en cuyo caso predecimos una cadena vacia).

**TensorFlow/Keras:**

Ahora pongamos todo lo que acabamos de hacer en una funcion `compute_metrics()` que usaremos despues de entrenar nuestro modelo. Necesitaremos pasar un poco mas que solo los logits de salida, ya que tenemos que buscar en el conjunto de datos de caracteristicas los offsets y en el conjunto de datos de ejemplos los contextos originales:


```python
from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Iterar a traves de todas las caracteristicas asociadas con ese ejemplo
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Omitir respuestas que no estan completamente en el contexto
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Omitir respuestas con una longitud que es < 0 o > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Seleccionar la respuesta con la mejor puntuacion
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
```

Podemos verificar que funciona en nuestras predicciones:

```python
compute_metrics(start_logits, end_logits, eval_set, small_eval_set)
```

```python out
{'exact_match': 83.0, 'f1': 88.25}
```

Se ve bien! Ahora usemos esto para ajustar finamente nuestro modelo.

### Ajuste fino del modelo[[fine-tuning-the-model]]


**PyTorch:**

Ahora estamos listos para entrenar nuestro modelo. Primero creemoslo, usando la clase `AutoModelForQuestionAnswering` como antes:

```python
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

**TensorFlow/Keras:**

Ahora estamos listos para entrenar nuestro modelo. Primero creemoslo, usando la clase `TFAutoModelForQuestionAnswering` como antes:

```python
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```


Como es usual, obtenemos una advertencia de que algunos pesos no se usan (los de la cabeza de preentrenamiento) y algunos otros se inicializan aleatoriamente (los de la cabeza de respuesta a preguntas). Ya deberias estar acostumbrado a esto, pero eso significa que este modelo no esta listo para usarse todavia y necesita ajuste fino -- que bueno que estamos a punto de hacer eso!

Para poder subir nuestro modelo al Hub, necesitaremos iniciar sesion en Hugging Face. Si estas ejecutando este codigo en un notebook, puedes hacerlo con la siguiente funcion de utilidad, que muestra un widget donde puedes ingresar tus credenciales de inicio de sesion:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Si no estas trabajando en un notebook, simplemente escribe la siguiente linea en tu terminal:

```bash
huggingface-cli login
```


**PyTorch:**

Una vez hecho esto, podemos definir nuestros `TrainingArguments`. Como dijimos cuando definimos nuestra funcion para calcular la metrica, no podremos tener un ciclo de evaluacion regular debido a la firma de la funcion `compute_metrics()`. Podriamos escribir nuestra propia subclase de `Trainer` para hacer esto (un enfoque que puedes encontrar en el [script de ejemplo de respuesta a preguntas](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py)), pero eso es demasiado largo para esta seccion. En su lugar, solo evaluaremos el modelo al final del entrenamiento aqui y te mostraremos como hacer una evaluacion regular en "Un ciclo de entrenamiento personalizado" mas abajo.

Esto es realmente donde la API del `Trainer` muestra sus limites y la biblioteca Accelerate brilla: personalizar la clase para un caso de uso especifico puede ser doloroso, pero modificar un ciclo de entrenamiento completamente expuesto es facil.

Echemos un vistazo a nuestros `TrainingArguments`:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)
```

Hemos visto la mayoria de estos antes: establecemos algunos hiperparametros (como la tasa de aprendizaje, el numero de epocas de entrenamiento y algo de weight decay) e indicamos que queremos guardar el modelo al final de cada epoca, omitir la evaluacion y subir nuestros resultados al Model Hub. Tambien habilitamos el entrenamiento de precision mixta con `fp16=True`, ya que puede acelerar el entrenamiento de manera agradable en una GPU reciente.

**TensorFlow/Keras:**

Ahora que eso esta hecho, podemos crear nuestros TF Datasets. Podemos usar el data collator simple por defecto esta vez:

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")
```

Y ahora creamos los datasets como es usual.

```python
tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)
tf_eval_dataset = model.prepare_tf_dataset(
    validation_dataset,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```

A continuacion, configuramos nuestros hiperparametros de entrenamiento y compilamos nuestro modelo:

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# El numero de pasos de entrenamiento es el numero de muestras en el dataset, dividido por el tamano del lote y luego multiplicado
# por el numero total de epocas. Ten en cuenta que el tf_train_dataset aqui es un tf.data.Dataset en lotes,
# no el Dataset original de Hugging Face, por lo que su len() ya es num_samples // batch_size.
num_train_epochs = 3
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Entrenar en precision mixta float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Finalmente, estamos listos para entrenar con `model.fit()`. Usamos un `PushToHubCallback` para subir el modelo al Hub despues de cada epoca.


Por defecto, el repositorio usado estara en tu namespace y se llamara como el directorio de salida que estableciste, asi que en nuestro caso estara en `"sgugger/bert-finetuned-squad"`. Podemos anular esto pasando un `hub_model_id`; por ejemplo, para subir el modelo a la organizacion `huggingface_course` usamos `hub_model_id="huggingface_course/bert-finetuned-squad"` (que es el modelo que enlazamos al principio de esta seccion).


**PyTorch:**

> [!TIP]
> Si el directorio de salida que estas usando existe, necesita ser un clon local del repositorio al que quieres subir (asi que establece un nuevo nombre si obtienes un error al definir tu `Trainer`).

Finalmente, simplemente pasamos todo a la clase `Trainer` y lanzamos el entrenamiento:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**TensorFlow/Keras:**

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-finetuned-squad", tokenizer=tokenizer)

# Vamos a hacer la validacion despues, asi que no hay validacion durante el entrenamiento
model.fit(tf_train_dataset, callbacks=[callback], epochs=num_train_epochs)
```


Ten en cuenta que mientras el entrenamiento sucede, cada vez que el modelo se guarda (aqui, cada epoca) se sube al Hub en segundo plano. De esta manera, podras reanudar tu entrenamiento en otra maquina si es necesario. Todo el entrenamiento toma un rato (un poco mas de una hora en una Titan RTX), asi que puedes tomar un cafe o releer algunas de las partes del curso que encontraste mas desafiantes mientras procede. Tambien ten en cuenta que tan pronto como la primera epoca termine, veras algunos pesos subidos al Hub y podras empezar a jugar con tu modelo en su pagina.


**PyTorch:**

Una vez que el entrenamiento este completo, finalmente podemos evaluar nuestro modelo (y rezar para que no hayamos gastado todo ese tiempo de computo en nada). El metodo `predict()` del `Trainer` devolvera una tupla donde los primeros elementos seran las predicciones del modelo (aqui un par con los logits de inicio y fin). Enviamos esto a nuestra funcion `compute_metrics()`:

```python
predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
```

**TensorFlow/Keras:**

Una vez que el entrenamiento este completo, finalmente podemos evaluar nuestro modelo (y rezar para que no hayamos gastado todo ese tiempo de computo en nada). El metodo `predict()` de nuestro `model` se encargara de obtener las predicciones, y como hicimos todo el trabajo duro de definir una funcion `compute_metrics()` antes, podemos obtener nuestros resultados en una sola linea:

```python
predictions = model.predict(tf_eval_dataset)
compute_metrics(
    predictions["start_logits"],
    predictions["end_logits"],
    validation_dataset,
    raw_datasets["validation"],
)
```


```python out
{'exact_match': 81.18259224219489, 'f1': 88.67381321905516}
```

Genial! Como comparacion, las puntuaciones base reportadas en el articulo de BERT para este modelo son 80.8 y 88.5, asi que estamos justo donde deberiamos estar.


**PyTorch:**

Finalmente, usamos el metodo `push_to_hub()` para asegurarnos de subir la ultima version del modelo:

```py
trainer.push_to_hub(commit_message="Training complete")
```

Esto devuelve la URL del commit que acaba de hacer, si quieres inspeccionarlo:

```python out
'https://huggingface.co/sgugger/bert-finetuned-squad/commit/9dcee1fbc25946a6ed4bb32efb1bd71d5fa90b68'
```

El `Trainer` tambien redacta una tarjeta de modelo con todos los resultados de evaluacion y la sube.


En este punto, puedes usar el widget de inferencia en el Model Hub para probar el modelo y compartirlo con tus amigos, familia y mascotas favoritas. Has ajustado finamente con exito un modelo en una tarea de respuesta a preguntas -- felicitaciones!

> [!TIP]
> Tu turno! Prueba otra arquitectura de modelo para ver si funciona mejor en esta tarea!


**PyTorch:**

Si quieres profundizar un poco mas en el ciclo de entrenamiento, ahora te mostraremos como hacer lo mismo usando Accelerate.

## Un ciclo de entrenamiento personalizado[[a-custom-training-loop]]

Ahora echemos un vistazo al ciclo de entrenamiento completo, para que puedas personalizar facilmente las partes que necesites. Se parecera mucho al ciclo de entrenamiento en el [Capitulo 3](/course/chapter3/4), con la excepcion del ciclo de evaluacion. Podremos evaluar el modelo regularmente ya que no estamos restringidos por la clase `Trainer`.

### Preparando todo para el entrenamiento[[preparing-everything-for-training]]

Primero necesitamos construir los `DataLoader`s a partir de nuestros datasets. Establecemos el formato de esos datasets a `"torch"`, y removemos las columnas en el conjunto de validacion que no son usadas por el modelo. Luego, podemos usar el `default_data_collator` proporcionado por Transformers como un `collate_fn` y mezclar el conjunto de entrenamiento, pero no el de validacion:

```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8
)
```

A continuacion reinstanciamos nuestro modelo, para asegurarnos de que no estamos continuando el ajuste fino de antes sino empezando desde el modelo BERT preentrenado de nuevo:

```py
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

Luego necesitaremos un optimizador. Como es usual usamos el clasico `AdamW`, que es como Adam, pero con una correccion en la forma en que se aplica el weight decay:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Una vez que tenemos todos esos objetos, podemos enviarlos al metodo `accelerator.prepare()`. Recuerda que si quieres entrenar en TPUs en un notebook de Colab, necesitaras mover todo este codigo a una funcion de entrenamiento, y esa no deberia ejecutar ninguna celda que instancie un `Accelerator`. Podemos forzar el entrenamiento de precision mixta pasando `fp16=True` al `Accelerator` (o, si estas ejecutando el codigo como un script, solo asegurate de llenar la `config` de Accelerate apropiadamente).

```py
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Como deberias saber de las secciones anteriores, solo podemos usar la longitud del `train_dataloader` para calcular el numero de pasos de entrenamiento despues de que haya pasado por el metodo `accelerator.prepare()`. Usamos el mismo schedule lineal que en las secciones anteriores:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Para subir nuestro modelo al Hub, necesitaremos crear un objeto `Repository` en una carpeta de trabajo. Primero inicia sesion en el Hub de Hugging Face, si no has iniciado sesion ya. Determinaremos el nombre del repositorio a partir del ID del modelo que queremos darle a nuestro modelo (sientete libre de reemplazar el `repo_name` con tu propia eleccion; solo necesita contener tu nombre de usuario, que es lo que hace la funcion `get_full_repo_name()`):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
'sgugger/bert-finetuned-squad-accelerate'
```

Luego podemos clonar ese repositorio en una carpeta local. Si ya existe, esta carpeta local deberia ser un clon del repositorio con el que estamos trabajando:

```py
output_dir = "bert-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

Ahora podemos subir cualquier cosa que guardemos en `output_dir` llamando al metodo `repo.push_to_hub()`. Esto nos ayudara a subir los modelos intermedios al final de cada epoca.

## Ciclo de entrenamiento[[training-loop]]

Ahora estamos listos para escribir el ciclo de entrenamiento completo. Despues de definir una barra de progreso para seguir como va el entrenamiento, el ciclo tiene tres partes:

- El entrenamiento en si, que es la iteracion clasica sobre el `train_dataloader`, paso hacia adelante a traves del modelo, luego paso hacia atras y paso del optimizador.
- La evaluacion, en la que recopilamos todos los valores para `start_logits` y `end_logits` antes de convertirlos a arrays de NumPy. Una vez que el ciclo de evaluacion termina, concatenamos todos los resultados. Ten en cuenta que necesitamos truncar porque el `Accelerator` puede haber agregado algunas muestras al final para asegurar que tenemos el mismo numero de ejemplos en cada proceso.
- Guardar y subir, donde primero guardamos el modelo y el tokenizador, luego llamamos a `repo.push_to_hub()`. Como hicimos antes, usamos el argumento `blocking=False` para decirle a la biblioteca Hub que suba en un proceso asincrono. De esta manera, el entrenamiento continua normalmente y esta instruccion (larga) se ejecuta en segundo plano.

Aqui esta el codigo completo para el ciclo de entrenamiento:

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Entrenamiento
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluacion
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    print(f"epoch {epoch}:", metrics)

    # Guardar y subir
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
```

En caso de que esta sea la primera vez que ves un modelo guardado con Accelerate, tomemos un momento para inspeccionar las tres lineas de codigo que lo acompanan:

```py
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

La primera linea es autoexplicativa: le dice a todos los procesos que esperen hasta que todos esten en esa etapa antes de continuar. Esto es para asegurarnos de que tenemos el mismo modelo en cada proceso antes de guardar. Luego tomamos el `unwrapped_model`, que es el modelo base que definimos. El metodo `accelerator.prepare()` cambia el modelo para trabajar en entrenamiento distribuido, por lo que ya no tendra el metodo `save_pretrained()`; el metodo `accelerator.unwrap_model()` deshace ese paso. Por ultimo, llamamos a `save_pretrained()` pero le decimos a ese metodo que use `accelerator.save()` en lugar de `torch.save()`.

Una vez que esto este hecho, deberias tener un modelo que produce resultados bastante similares al entrenado con el `Trainer`. Puedes revisar el modelo que entrenamos usando este codigo en [*huggingface-course/bert-finetuned-squad-accelerate*](https://huggingface.co/huggingface-course/bert-finetuned-squad-accelerate). Y si quieres probar cualquier ajuste al ciclo de entrenamiento, puedes implementarlos directamente editando el codigo mostrado arriba!


## Usando el modelo ajustado finamente[[using-the-fine-tuned-model]]

Ya te hemos mostrado como puedes usar el modelo que ajustamos finamente en el Model Hub con el widget de inferencia. Para usarlo localmente en un `pipeline`, solo tienes que especificar el identificador del modelo:

```py
from transformers import pipeline

# Reemplaza esto con tu propio checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
Transformers is backed by the three most popular deep learning libraries - Jax, PyTorch and TensorFlow - with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back Transformers?"
question_answerer(question=question, context=context)
```

```python out
{'score': 0.9979003071784973,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}
```

Genial! Nuestro modelo esta funcionando tan bien como el predeterminado para este pipeline!


---

# Dominando los LLMs[[mastering-llms]]


Si has llegado hasta aqui en el curso, felicidades -- ahora tienes todo el conocimiento y las herramientas que necesitas para abordar (casi) cualquier tarea de lenguaje con 🤗 Transformers y el ecosistema de Hugging Face!

## De NLP a LLMs

Aunque hemos cubierto muchas tareas tradicionales de NLP en este curso, el campo ha sido revolucionado por los Modelos de Lenguaje Grande (LLMs). Estos modelos han expandido dramaticamente lo que es posible en el procesamiento de lenguaje:

- Pueden manejar multiples tareas sin ajuste fino especifico de tarea
- Sobresalen siguiendo instrucciones y adaptandose a diferentes contextos
- Pueden generar texto coherente y contextualmente apropiado para varias aplicaciones
- Pueden realizar razonamiento y resolver problemas complejos a traves de tecnicas como chain-of-thought prompting

Las habilidades fundamentales de NLP que has aprendido siguen siendo esenciales para trabajar con LLMs efectivamente. Entender la tokenizacion, arquitecturas de modelos, enfoques de ajuste fino y metricas de evaluacion proporciona el conocimiento necesario para aprovechar los LLMs a su maximo potencial.

Hemos visto muchos data collators diferentes, asi que hicimos este pequeno video para ayudarte a encontrar cual usar para cada tarea:


**Video:** [Ver en YouTube](https://youtu.be/-RPeakdlHYo)


Despues de completar este recorrido relampago por las tareas principales de lenguaje, deberias:

* Saber que arquitecturas (encoder, decoder, o encoder-decoder) son mas adecuadas para cada tarea
* Entender la diferencia entre preentrenar y ajustar finamente un modelo de lenguaje
* Saber como entrenar modelos Transformer usando ya sea la API `Trainer` y las caracteristicas de entrenamiento distribuido de 🤗 Accelerate o TensorFlow y Keras, dependiendo de que track hayas estado siguiendo
* Entender el significado y limitaciones de metricas como ROUGE y BLEU para tareas de generacion de texto
* Saber como interactuar con tus modelos ajustados finamente, tanto en el Hub como usando el `pipeline` de 🤗 Transformers
* Apreciar como los LLMs se construyen sobre y extienden las tecnicas tradicionales de NLP

A pesar de todo este conocimiento, llegara un momento en que encontraras un bug dificil en tu codigo o tendras una pregunta sobre como resolver un problema particular de procesamiento de lenguaje. Afortunadamente, la comunidad de Hugging Face esta aqui para ayudarte! En el capitulo final de esta parte del curso, exploraremos como depurar tus modelos Transformer y pedir ayuda efectivamente.


---




# Cuestionario de fin de capitulo[[end-of-chapter-quiz]]


Probemos lo que aprendiste en este capitulo!

### 1. Cuales de las siguientes tareas pueden enmarcarse como un problema de clasificacion de tokens?


- Encontrar los componentes gramaticales en una oracion.
- Encontrar si una oracion es gramaticalmente correcta o no.
- Encontrar las personas mencionadas en una oracion.
- Encontrar el fragmento de palabras en una oracion que responde a una pregunta.


### 2. Que parte del preprocesamiento para clasificacion de tokens difiere de los otros pipelines de preprocesamiento?


- No hay necesidad de hacer nada; los textos ya estan tokenizados.
- Los textos se dan como palabras, asi que solo necesitamos aplicar tokenizacion por subpalabras.
- Usamos `-100` para etiquetar los tokens especiales.
- Necesitamos asegurarnos de truncar o rellenar las etiquetas al mismo tamano que las entradas, cuando aplicamos truncamiento/relleno.


### 3. Que problema surge cuando tokenizamos las palabras en un problema de clasificacion de tokens y queremos etiquetar los tokens?


- El tokenizador agrega tokens especiales y no tenemos etiquetas para ellos.
- Cada palabra puede producir varios tokens, por lo que terminamos con mas tokens de los que tenemos etiquetas.
- Los tokens agregados no tienen etiquetas, asi que no hay problema.


### 4. Que significa "adaptacion de dominio"?


- Es cuando ejecutamos un modelo en un conjunto de datos y obtenemos las predicciones para cada muestra en ese conjunto.
- Es cuando entrenamos un modelo en un conjunto de datos.
- Es cuando ajustamos finamente un modelo preentrenado en un nuevo conjunto de datos, y da predicciones que estan mas adaptadas a ese conjunto de datos.
- Es cuando agregamos muestras mal clasificadas a un conjunto de datos para hacer nuestro modelo mas robusto.


### 5. Cuales son las etiquetas en un problema de modelado de lenguaje enmascarado?


- Algunos de los tokens en la oracion de entrada se enmascaran aleatoriamente y las etiquetas son los tokens de entrada originales.
- Algunos de los tokens en la oracion de entrada se enmascaran aleatoriamente y las etiquetas son los tokens de entrada originales, desplazados a la izquierda.
- Algunos de los tokens en la oracion de entrada se enmascaran aleatoriamente, y la etiqueta es si la oracion es positiva o negativa.
- Algunos de los tokens en las dos oraciones de entrada se enmascaran aleatoriamente, y la etiqueta es si las dos oraciones son similares o no.


### 6. Cuales de estas tareas pueden verse como un problema de secuencia a secuencia?


- Escribir resenas cortas de documentos largos
- Responder preguntas sobre un documento
- Traducir un texto en chino al ingles
- Arreglar los mensajes enviados por mi sobrino/amigo para que esten en espanol correcto


### 7. Cual es la forma correcta de preprocesar los datos para un problema de secuencia a secuencia?


- Las entradas y objetivos deben enviarse juntos al tokenizador con `inputs=...` y `targets=...`.
- Las entradas y los objetivos deben preprocesarse, en dos llamadas separadas al tokenizador.
- Como de costumbre, solo tenemos que tokenizar las entradas.
- Las entradas deben enviarse al tokenizador, y los objetivos tambien, pero bajo un administrador de contexto especial.


**PyTorch:**

### 8. Por que hay una subclase especifica de `Trainer` para problemas de secuencia a secuencia?


- Porque los problemas de secuencia a secuencia usan una perdida personalizada, para ignorar las etiquetas establecidas en `-100`
- Porque los problemas de secuencia a secuencia requieren un bucle de evaluacion especial
- Porque los objetivos son textos en problemas de secuencia a secuencia
- Porque usamos dos modelos en problemas de secuencia a secuencia


**TensorFlow/Keras:**

### 9. Por que a menudo es innecesario especificar una perdida cuando se llama a `compile()` en un modelo Transformer?


- Porque los modelos Transformer se entrenan con aprendizaje no supervisado
- Porque la salida de perdida interna del modelo se usa por defecto
- Porque calculamos metricas despues del entrenamiento en su lugar
- Porque la perdida se especifica en `model.fit()` en su lugar


### 10. Cuando deberias preentrenar un nuevo modelo?


- Cuando no hay un modelo preentrenado disponible para tu idioma especifico
- Cuando tienes muchos datos disponibles, incluso si hay un modelo preentrenado que podria funcionar en ellos
- Cuando tienes preocupaciones sobre el sesgo del modelo preentrenado que estas usando
- Cuando los modelos preentrenados disponibles simplemente no son lo suficientemente buenos


### 11. Por que es facil preentrenar un modelo de lenguaje con muchos y muchos textos?


- Porque hay muchos textos disponibles en internet
- Porque el objetivo de preentrenamiento no requiere que humanos etiqueten los datos
- Porque la biblioteca 🤗 Transformers solo requiere unas pocas lineas de codigo para comenzar el entrenamiento


### 12. Cuales son los principales desafios al preprocesar datos para una tarea de respuesta a preguntas?


- Necesitas tokenizar las entradas.
- Necesitas lidiar con contextos muy largos, que dan varias caracteristicas de entrenamiento que pueden o no tener la respuesta en ellas.
- Necesitas tokenizar las respuestas a la pregunta asi como las entradas.
- Del span de respuesta en el texto, tienes que encontrar los tokens de inicio y fin en la entrada tokenizada.


### 13. Como se hace usualmente el post-procesamiento en respuesta a preguntas?


- El modelo te da las posiciones de inicio y fin de la respuesta, y solo tienes que decodificar el span correspondiente de tokens.
- El modelo te da las posiciones de inicio y fin de la respuesta para cada caracteristica creada por un ejemplo, y solo tienes que decodificar el span correspondiente de tokens en la que tiene la mejor puntuacion.
- El modelo te da las posiciones de inicio y fin de la respuesta para cada caracteristica creada por un ejemplo, y solo tienes que hacerlas coincidir con el span en el contexto para la que tiene la mejor puntuacion.
- El modelo genera una respuesta, y solo tienes que decodificarla.



---

# 8. ¿Cómo solicitar ayuda?

# Introducción


Ahora sabes cómo abordar las tareas de PLN más comunes con la librería 🤗 Transformers, ¡deberías ser capaz de iniciar tus propios proyectos! En este capítulo exploraremos qué debes hacer cuando te encuentras con un problema. Aprenderás a cómo depurar (debug) exitosamente tu código o tu entrenamiento, y cómo solicitar ayuda si no consigues resolver el problema por ti mismo. Además, si crees que has encontrado un error (bug) en una de las librerías de Hugging Face, te indicaremos la mejor manera de reportarlo para que se resuelva tan pronto como sea posible.

Más precisamente, en este capítulo aprenderás:

- Lo primero que debes hacer cuando se produce un error
- Cómo solicitar ayuda en los [foros](https://discuss.huggingface.co/)
- Cómo depurar tu pipeline de entrenamiento
- Cómo escribir un buen issue

Nada de esto es específicamente relacionado con la librería 🤗 Transformers o con el ecosistema de Hugging Face, por supuesto; ¡las lecciones de este capítulo son aplicables a la mayoría de proyectos de open source!


---

# ¿Qué hacer cuando se produce un error?


En esta sección veremos algunos errores comunes que pueden ocurrir cuando intentas generar predicciones a partir de tu modelo Transformer recién afinado. Esto te preparará para la [sección 4](/course/chapter8/section4), en la que exploraremos cómo depurar (debug) la fase de entrenamiento.


**Video:** [Ver en YouTube](https://youtu.be/DQ-CpJn6Rc4)


Hemos preparado un [repositorio de un modelo de ejemplo](https://huggingface.co/lewtun/distilbert-base-uncased-finetuned-squad-d5716d28) para esta sección, por lo que si deseas ejecutar el código en este capítulo, primero necesitarás copiar el modelo a tu cuenta en el [Hub de Hugging Face](https://huggingface.co). Para ello, primero inicia sesión (log in) ejecutando lo siguiente en una Jupyter notebook:

```python
from huggingface_hub import notebook_login

notebook_login()
```

o puedes ejecutar lo siguiente en tu terminal favorita:

```bash
huggingface-cli login
```

Esto te pedirá que introduzcas tu nombre de usuario y contraseña, y guardará un token en *~/.cache/huggingface/*. Una vez que hayas iniciado sesión, puedes copiar el repositorio de ejemplo con la siguiente función:

```python
from distutils.dir_util import copy_tree
from huggingface_hub import Repository, snapshot_download, create_repo, get_full_repo_name


def copy_repository_template():
    # Clona el repo y extrae la ruta local
    template_repo_id = "lewtun/distilbert-base-uncased-finetuned-squad-d5716d28"
    commit_hash = "be3eaffc28669d7932492681cd5f3e8905e358b4"
    template_repo_dir = snapshot_download(template_repo_id, revision=commit_hash)
    # Crea un repo vacío en el Hub
    model_name = template_repo_id.split("/")[1]
    create_repo(model_name, exist_ok=True)
    # Clona el repo vacío
    new_repo_id = get_full_repo_name(model_name)
    new_repo_dir = model_name
    repo = Repository(local_dir=new_repo_dir, clone_from=new_repo_id)
    # Copia los archivos
    copy_tree(template_repo_dir, new_repo_dir)
    # Envia (push) al Hub
    repo.push_to_hub()
```

Ahora cuando llames a la función `copy_repository_template()`, esta creará una copia del repositorio de ejemplo en tu cuenta.

## Depurando el pipeline de 🤗 Transformers

Para iniciar nuestro viaje hacia el maravilloso mundo de la depuración de modelos de Transformers, imagina lo siguiente: estás trabajando con un compañero en un proyecto de respuesta a preguntas (question answering) para ayudar a los clientes de un sitio web de comercio electrónico a encontrar respuestas sobre productos de consumo. Tu compañero te envía el siguiente mensaje: 

> ¡Buen día! Acabo de lanzar un experimento usando las técnicas del [Capitulo 7](/course/chapter7/7) del curso de Hugging Face y ¡obtuvo unos buenos resultados con el conjunto de datos SQuAD! Creo que podemos usar este modelo como punto de partida para nuestro proyecto. El identificador del modelo en el Hub es "lewtun/distillbert-base-uncased-finetuned-squad-d5716d28". No dudes en probarlo :)

y en lo primero que piensas es en cargar el modelo usando el `pipeline` de la librería 🤗 Transformers:

```python
from transformers import pipeline

model_checkpoint = get_full_repo_name("distillbert-base-uncased-finetuned-squad-d5716d28")
reader = pipeline("question-answering", model=model_checkpoint)
```

```python out
"""
OSError: Can't load config for 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28'. Make sure that:

- 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is the correct path to a directory containing a config.json file
"""
```

¡Oh no, algo parece estar mal! Si eres nuevo en programación, este tipo de errores pueden parecer un poco crípticos al inicio (¿qué es un `OSError`?). El error mostrado aquí es solo la última parte de un reporte de errores mucho más largo llamado _Python traceback_ (o _stack trace_). Por ejemplo, si estás ejecutando este código en Google Colab, podrías ver algo parecido como la siguiente captura:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/traceback.png" alt="A Python traceback." width="100%"/>
</div>

Hay mucha información contenida en estos reportes, así que vamos a repasar juntos las partes clave. La primera cosa que notamos es que el _traceback_ debería ser leído de _abajo hacia arriba_. Esto puede sonar extraño si estás acostumbrado a leer en español de arriba hacia abajo, pero refleja el hecho de que el _traceback_ muestra la secuencia de funciones llamadas que el `pipeline` realiza al descargar el modelo y el tokenizador. (Ve al [Capítulo 2](/course/chapter2) para más detalles sobre cómo funciona el `pipeline` bajo el capó) 

> [!TIP]
> 🚨 ¿Ves el cuadro azul alrededor de "6 frames" en el traceback de Google Colab? Es una característica especial de Colab, que comprime el traceback en "frames". Si no puedes encontrar el origen de un error, asegúrate de ampliar el traceback completo haciendo clic en esas dos flechitas.

Esto significa que la última línea del traceback indica el último mensaje de error y nos da el nombre de la excepción (exception) que se ha generado. En este caso, el tipo de excepción es `OSError`, lo que indica un error relacionado con el sistema. Si leemos el mensaje de error que lo acompaña, podemos ver que parece haber un problema con el archivo *config.json* del modelo, y nos da dos sugerencias para solucionarlo:

```python out
"""
Make sure that:

- 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is the correct path to a directory containing a config.json file
"""
```

> [!TIP]
> 💡 Si te encuentras con un mensaje de error difícil de entender, simplemente copia y pega el mensaje en la barra de búsqueda de Google o de [Stack Overflow](https://stackoverflow.com/) (¡sí, en serio!). Es muy posible que no seas la primera persona en encontrar el error, y esta es una buena forma de hallar soluciones que otros miembros de la comunidad han publicado. Por ejemplo, al buscar `OSError: Can't load config for` en Stack Overflow se obtienen varios resultados que pueden ser utilizados como punto de partida para resolver el problema.

La primera sugerencia nos pide que comprobemos si el identificador del modelo es realmente correcto, así que lo primero es copiar el identificador y pegarlo en la barra de búsqueda del Hub:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/wrong-model-id.png" alt="The wrong model name." width="100%"/>
</div>

Hmm, efectivamente parece que el modelo de nuestro compañero no está en el Hub... ¡pero hay una errata en el nombre del modelo! DistilBERT solo tiene una "l" en el nombre, así que vamos a corregirlo y a buscar "lewtun/distilbert-base-uncased-finetuned-squad-d5716d28" en su lugar:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/true-model-id.png" alt="The right model name." width="100%"/>
</div>

Bien, esto dio resultado. Ahora vamos a intentar descargar el modelo de nuevo con el identificador correcto:

```python
model_checkpoint = get_full_repo_name("distilbert-base-uncased-finetuned-squad-d5716d28")
reader = pipeline("question-answering", model=model_checkpoint)
```

```python out
"""
OSError: Can't load config for 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28'. Make sure that:

- 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28' is the correct path to a directory containing a config.json file
"""
```

Argh, falló de nuevo, ¡bienvenido al día a día de un ingeniero de machine learning! Dado que arreglamos el identificador del modelo, el problema debe estar en el repositorio. Una manera rápida de acceder a los contenidos de un repositorio en el 🤗 Hub es por medio de la función `list_repo_files()` de la librería de `huggingface_hub`:

```python
from huggingface_hub import list_repo_files

list_repo_files(repo_id=model_checkpoint)
```

```python out
['.gitattributes', 'README.md', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json', 'training_args.bin', 'vocab.txt']
```

Interesante. ¡No parece haber un archivo *config.json* en el repositorio! No es de extrañar que nuestro `pipeline` no pudiera cargar el modelo; nuestro compañero debe haberse olvidado de enviar este archivo al Hub después de ajustarlo (fine-tuned). En este caso, el problema parece bastante simple de resolver: podemos pedirle a nuestro compañero que añada el archivo, o, ya que podemos ver en el identificador del modelo que el modelo preentrenado fue [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), podemos descargar la configuración para este modelo y enviarla a nuestro repositorio para ver si eso resuelve el problema. Intentemos esto. Usando las técnicas que aprendimos en el [Capítulo 2](/course/chapter2), podemos descargar la configuración del modelo con la clase `AutoConfig`:

```python
from transformers import AutoConfig

pretrained_checkpoint = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(pretrained_checkpoint)
```

> [!WARNING]
> 🚨 El enfoque que tomamos aquí no es infalible, ya que nuestro compañero puede haber cambiado la configuración de `distilbert-base-uncased` antes de ajustar (fine-tuning) el modelo. En la vida real, nos gustaría consultar con él primero, pero para los fines de esta sección asumiremos que usó la configuración predeterminada.

Luego podemos enviar esto a nuestro repositorio del modelo con la función de configuración `push_to_hub()`: 

```python
config.push_to_hub(model_checkpoint, commit_message="Add config.json")
```

Ahora podemos probar si esto funciona cargando el modelo desde el último commit de la rama `main`: 

```python
reader = pipeline("question-answering", model=model_checkpoint, revision="main")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text
given a question. An example of a question answering dataset is the SQuAD
dataset, which is entirely based on that task. If you would like to fine-tune a
model on a SQuAD task, you may leverage the
examples/pytorch/question-answering/run_squad.py script.

🤗 Transformers is interoperable with the PyTorch, TensorFlow, and JAX
frameworks, so you can use your favourite tools for a wide variety of tasks!
"""

context_es = r"""
La respuesta a preguntas es la extracción de una respuesta textual a partir de 
una pregunta. Un ejemplo de conjunto de datos de respuesta a preguntas es el 
dataset SQuAD, que se basa por completo en esta tarea. Si deseas afinar un modelo 
en una tarea SQuAD, puedes aprovechar el script
 examples/pytorch/question-answering/run_squad.py

🤗 Transformers es interoperable con los frameworks PyTorch, TensorFlow y JAX, 
así que ¡puedes utilizar tus herramientas favoritas para una gran variedad de tareas!
"""

question = "What is extractive question answering?"
# ¿Qué es la respuesta extractiva a preguntas?
reader(question=question, context=context)
```

```python out
{'score': 0.38669535517692566,
 'start': 34,
 'end': 95,
 'answer': 'the task of extracting an answer from a text given a question'}
 # la tarea de extraer una respuesta de un texto a una pregunta dada
```

¡Yuju, funcionó! Recapitulemos lo que acabas de aprender:

- Los mensajes de error en Python son conocidos como _tracebacks_ y se leen de abajo hacia arriba. La última línea del mensaje de error generalmente contiene la información que necesitas para ubicar la fuente del problema.
- Si la última línea no contiene suficiente información, sigue el traceback y mira si puedes identificar en qué parte del código fuente se produce el error.
- Si ninguno de los mensajes de error te ayuda a depurar el problema, trata de buscar en internet una solución a un problema similar.
- El 🤗 `huggingface_hub` de la librería proporciona un conjunto de herramientas que puedes utilizar para interactuar y depurar los repositorios en el Hub. 

Ahora que sabes cómo depurar un pipeline, vamos a ver un ejemplo más complicado en la pasada hacia delante (forward pass) del propio modelo.

## Depurando la pasada hacia delante (forward pass) de tu modelo

Aunque el `pipeline` es estupendo para la mayoría de las aplicaciones en las que necesitas generar predicciones rápidamente, a veces necesitarás acceder a los _logits_ del modelo (por ejemplo, si tienes algún postprocesamiento personalizado que te gustaría aplicar). Para ver lo que puede salir mal en este caso, vamos a coger primero el modelo y el tokenizador de nuestro `pipeline`:

```python
tokenizer = reader.tokenizer
model = reader.model
```

A continuación, necesitamos una pregunta, así que veamos si nuestros frameworks son compatibles:

```python
question = "Which frameworks can I use?"  # ¿Qué frameworks puedo usar?
```

Como vimos en el [Capítulo 7](/course/chapter7), los pasos habituales que debemos seguir son tokenizar los inputs, extraer los _logits_ de los tokens de inicio y fin y luego decodificar el intervalo de la respuesta: 

```python
import torch

inputs = tokenizer(question, context, add_special_tokens=True)
input_ids = inputs["input_ids"][0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
# Obtiene el comienzo más probable de la respuesta con el argmax de la puntuación
answer_start = torch.argmax(answer_start_scores)
# Obtiene el final más probable de la respuesta con el argmax de la puntuación
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

```python out
"""
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/var/folders/28/k4cy5q7s2hs92xq7_h89_vgm0000gn/T/ipykernel_75743/2725838073.py in <module>
      1 inputs = tokenizer(question, text, add_special_tokens=True)
      2 input_ids = inputs["input_ids"]
----> 3 outputs = model(**inputs)
      4 answer_start_scores = outputs.start_logits
      5 answer_end_scores = outputs.end_logits

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, start_positions, end_positions, output_attentions, output_hidden_states, return_dict)
    723         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    724
--> 725         distilbert_output = self.distilbert(
    726             input_ids=input_ids,
    727             attention_mask=attention_mask,

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    471             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    472         elif input_ids is not None:
--> 473             input_shape = input_ids.size()
    474         elif inputs_embeds is not None:
    475             input_shape = inputs_embeds.size()[:-1]

AttributeError: 'list' object has no attribute 'size'
"""
```

Vaya, parece que tenemos un _bug_ en nuestro código. Pero no nos asusta un poco de depuración. Puedes usar el depurador de Python en una notebook:


**Video:** [Ver en YouTube](https://youtu.be/rSPyvPw0p9k)


o en una terminal: 


**Video:** [Ver en YouTube](https://youtu.be/5PkZ4rbHL6c)


Aquí la lectura del mensaje de error nos dice que el objeto `'list'` no tiene atributo `'size'`, y podemos ver una flecha `-->` apuntando a la línea donde el problema se originó en `model(**inputs)`. Puedes depurar esto interactivamente usando el _debugger_ de Python, pero por ahora simplemente imprimiremos un fragmento de `inputs` para ver qué obtenemos:

```python
inputs["input_ids"][:5]
```

```python out
[101, 2029, 7705, 2015, 2064]
```

Esto sin duda parece una `lista` ordinaria de Python, pero vamos a comprobar el tipo:

```python
type(inputs["input_ids"])
```

```python out
list
```

Sí, es una lista de Python. Entonces, ¿qué salió mal? Recordemos del [Capítulo 2](/course/chapter2) que las clases `AutoModelForXxx` en 🤗 Transformers operan con _tensores_ (tanto en PyTorch como en TensorFlow), y una operación común es extraer las dimensiones de un tensor usando `Tensor.size()` en, por ejemplo, PyTorch. Volvamos a echar un vistazo al traceback, para ver qué línea desencadenó la excepción:

```
~/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    471             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    472         elif input_ids is not None:
--> 473             input_shape = input_ids.size()
    474         elif inputs_embeds is not None:
    475             input_shape = inputs_embeds.size()[:-1]

AttributeError: 'list' object has no attribute 'size'
```

Parece que nuestro código trata de llamar a la función `input_ids.size()`, pero esta claramente no funcionará con una lista de Python, la cual solo es un contenedor. ¿Cómo podemos resolver este problema? La búsqueda del mensaje de error en Stack Overflow da bastantes [resultados](https://stackoverflow.com/search?q=AttributeError%3A+%27list%27+object+has+no+attribute+%27size%27&s=c15ec54c-63cb-481d-a749-408920073e8f) relevantes. Al hacer clic en el primero, aparece una pregunta similar a la nuestra, con la respuesta que se muestra en la siguiente captura de pantalla:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/stack-overflow.png" alt="An answer from Stack Overflow." width="100%"/>
</div>

La respuesta recomienda que adicionemos `return_tensors='pt'` al tokenizador, así que veamos si esto nos funciona:

```python out
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"][0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
# Obtiene el comienzo más probable de la respuesta con el argmax de la puntuación
answer_start = torch.argmax(answer_start_scores)
# Obtiene el final más probable de la respuesta con el argmax de la puntuación
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

```python out
"""
Question: Which frameworks can I use? # ¿Qué frameworks puedo usar?
Answer: pytorch, tensorflow, and jax
"""
```

¡Excelente, funcionó! Este en un gran ejemplo de lo útil que puede ser Stack Overflow: al identificar un problema similar, fuimos capaces de beneficiarnos de la experiencia de otros en la comunidad. Sin embargo, una búsqueda como esta no siempre dará una respuesta relevante, así que ¿qué podemos hacer en esos casos? Afortunadamente hay una comunidad acogedora de desarrolladores en los [foros de Hugging Face](https://discuss.huggingface.co/) que pueden ayudarte. En la siguiente sección, veremos cómo elaborar buenas preguntas en el foro que tengan posibilidades de ser respondidas.


---

# Pedir ayuda en los foros[[asking-for-help-on-the-forums]]


**Video:** [Ver en YouTube](https://youtu.be/S2EEG3JIt2A)


Los [foros de Hugging Face](https://discuss.huggingface.co) son un excelente lugar para obtener ayuda del equipo de código abierto y de la comunidad más amplia de Hugging Face. Así es como se ve la página principal en un día cualquiera:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums.png" alt="Los foros de Hugging Face." width="100%"/>
</div>

En el lado izquierdo puedes ver todas las categorías en las que se agrupan los diversos temas, mientras que el lado derecho muestra los temas más recientes. Un tema es una publicación que contiene un título, categoría y descripción; es bastante similar al formato de issues de GitHub que vimos cuando creamos nuestro propio dataset en el [Capítulo 5](/course/chapter5). Como su nombre sugiere, la categoría [Beginners](https://discuss.huggingface.co/c/beginners/5) está principalmente destinada a personas que recién comienzan con las bibliotecas y el ecosistema de Hugging Face. Cualquier pregunta sobre cualquiera de las bibliotecas es bienvenida allí, ya sea para depurar código o para pedir ayuda sobre cómo hacer algo. (Dicho esto, si tu pregunta se refiere a una biblioteca en particular, probablemente deberías dirigirte a la categoría correspondiente de esa biblioteca en el foro.)

De manera similar, las categorías [Intermediate](https://discuss.huggingface.co/c/intermediate/6) y [Research](https://discuss.huggingface.co/c/research/7) son para preguntas más avanzadas, por ejemplo sobre las bibliotecas o alguna nueva investigación interesante de NLP que te gustaría discutir.

Y naturalmente, también debemos mencionar la categoría [Course](https://discuss.huggingface.co/c/course/20), donde puedes hacer cualquier pregunta que tengas relacionada con el curso de Hugging Face.

Una vez que hayas seleccionado una categoría, estarás listo para escribir tu primer tema. Puedes encontrar algunas [directrices](https://discuss.huggingface.co/t/how-to-request-support/3128) en el foro sobre cómo hacerlo, y en esta sección veremos algunas características que conforman un buen tema.

## Escribir una buena publicación en el foro[[writing-a-good-forum-post]]

Como ejemplo continuo, supongamos que estamos tratando de generar embeddings a partir de artículos de Wikipedia para crear un motor de búsqueda personalizado. Como de costumbre, cargamos el tokenizador y el modelo de la siguiente manera:

```python
from transformers import AutoTokenizer, AutoModel

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
```

Ahora supongamos que intentamos generar el embedding de una sección completa del [artículo de Wikipedia](https://en.wikipedia.org/wiki/Transformers) sobre Transformers (la franquicia, no la biblioteca):

```python
text = """
Generation One is a retroactive term for the Transformers characters that
appeared between 1984 and 1993. The Transformers began with the 1980s Japanese
toy lines Micro Change and Diaclone. They presented robots able to transform
into everyday vehicles, electronic items or weapons. Hasbro bought the Micro
Change and Diaclone toys, and partnered with Takara. Marvel Comics was hired by
Hasbro to create the backstory; editor-in-chief Jim Shooter wrote an overall
story, and gave the task of creating the characthers to writer Dennis O'Neil.
Unhappy with O'Neil's work (although O'Neil created the name "Optimus Prime"),
Shooter chose Bob Budiansky to create the characters.

The Transformers mecha were largely designed by Shōji Kawamori, the creator of
the Japanese mecha anime franchise Macross (which was adapted into the Robotech
franchise in North America). Kawamori came up with the idea of transforming
mechs while working on the Diaclone and Macross franchises in the early 1980s
(such as the VF-1 Valkyrie in Macross and Robotech), with his Diaclone mechs
later providing the basis for Transformers.

The primary concept of Generation One is that the heroic Optimus Prime, the
villainous Megatron, and their finest soldiers crash land on pre-historic Earth
in the Ark and the Nemesis before awakening in 1985, Cybertron hurtling through
the Neutral zone as an effect of the war. The Marvel comic was originally part
of the main Marvel Universe, with appearances from Spider-Man and Nick Fury,
plus some cameos, as well as a visit to the Savage Land.

The Transformers TV series began around the same time. Produced by Sunbow
Productions and Marvel Productions, later Hasbro Productions, from the start it
contradicted Budiansky's backstories. The TV series shows the Autobots looking
for new energy sources, and crash landing as the Decepticons attack. Marvel
interpreted the Autobots as destroying a rogue asteroid approaching Cybertron.
Shockwave is loyal to Megatron in the TV series, keeping Cybertron in a
stalemate during his absence, but in the comic book he attempts to take command
of the Decepticons. The TV series would also differ wildly from the origins
Budiansky had created for the Dinobots, the Decepticon turned Autobot Jetfire
(known as Skyfire on TV), the Constructicons (who combine to form
Devastator),[19][20] and Omega Supreme. The Marvel comic establishes early on
that Prime wields the Creation Matrix, which gives life to machines. In the
second season, the two-part episode The Key to Vector Sigma introduced the
ancient Vector Sigma computer, which served the same original purpose as the
Creation Matrix (giving life to Transformers), and its guardian Alpha Trion.
"""

inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits
```

```python output
IndexError: index out of range in self
```

Oh no, tenemos un problema -- y el mensaje de error es mucho más críptico que los que vimos en la [sección 2](/course/chapter8/section2). No podemos entender nada del traceback completo, así que decidimos acudir a los foros de Hugging Face en busca de ayuda. ¿Cómo podríamos redactar el tema?

Para comenzar, necesitamos hacer clic en el botón "New Topic" en la esquina superior derecha (ten en cuenta que para crear un tema, necesitaremos haber iniciado sesión):

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums-new-topic.png" alt="Creando un nuevo tema en el foro." width="100%"/>
</div>

Esto abre una interfaz de escritura donde podemos ingresar el título de nuestro tema, seleccionar una categoría y redactar el contenido:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic01.png" alt="La interfaz para crear un tema en el foro." width="100%"/>
</div>

Como el error parece ser exclusivamente sobre Transformers, seleccionaremos esta categoría. Nuestro primer intento de explicar el problema podría verse algo así:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic02.png" alt="Redactando el contenido para un nuevo tema en el foro." width="100%"/>
</div>

Aunque este tema contiene el mensaje de error con el que necesitamos ayuda, hay algunos problemas con la forma en que está escrito:

1. El título no es muy descriptivo, por lo que cualquier persona que navegue por el foro no podrá saber de qué trata el tema sin leer también el cuerpo.
2. El cuerpo no proporciona suficiente información sobre _de dónde_ proviene el error y _cómo_ reproducirlo.
3. El tema etiqueta directamente a algunas personas con un tono algo exigente.

Temas como este probablemente no obtendrán una respuesta rápida (si es que obtienen alguna), así que veamos cómo podemos mejorarlo. Comenzaremos con el primer problema de elegir un buen título.

### Elegir un título descriptivo[[choosing-a-descriptive-title]]

Si estás tratando de obtener ayuda con un error en tu código, una buena regla general es incluir suficiente información en el título para que otros puedan determinar rápidamente si creen que pueden responder tu pregunta o no. En nuestro ejemplo continuo, conocemos el nombre de la excepción que se está generando y tenemos algunas pistas de que se activa en el paso forward del modelo, donde llamamos a `model(**inputs)`. Para comunicar esto, un posible título podría ser:

> ¿Origen del IndexError en el paso forward de AutoModel?

Este título le dice al lector _dónde_ crees que está el error, y si han encontrado un `IndexError` antes, hay una buena probabilidad de que sepan cómo depurarlo. Por supuesto, el título puede ser lo que quieras, y otras variaciones como:

> ¿Por qué mi modelo produce un IndexError?

también podrían estar bien. Ahora que tenemos un título descriptivo, veamos cómo mejorar el cuerpo.

### Formatear tus fragmentos de código[[formatting-your-code-snippets]]

Leer código fuente ya es bastante difícil en un IDE, pero es aún más difícil cuando el código se copia y pega como texto plano. Afortunadamente, los foros de Hugging Face admiten el uso de Markdown, así que siempre debes encerrar tus bloques de código con tres acentos graves (```) para que sean más fáciles de leer. Hagamos esto para embellecer el mensaje de error -- y ya que estamos, hagamos el cuerpo un poco más cortés que nuestra versión original:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic03.png" alt="Nuestro tema del foro revisado, con formato de código apropiado." width="100%"/>
</div>

Como puedes ver en la captura de pantalla, encerrar los bloques de código en acentos graves convierte el texto sin formato en código formateado, completo con estilo de colores. También nota que se pueden usar acentos graves simples para formatear variables en línea, como hemos hecho para `distilbert-base-uncased`. Este tema se ve mucho mejor, y con un poco de suerte podríamos encontrar a alguien en la comunidad que pueda adivinar de qué se trata el error. Sin embargo, en lugar de depender de la suerte, hagamos la vida más fácil incluyendo el traceback en todo su detalle.

### Incluir el traceback completo[[including-the-full-traceback]]

Dado que la última línea del traceback suele ser suficiente para depurar tu propio código, puede ser tentador proporcionar solo eso en tu tema para "ahorrar espacio". Aunque bien intencionado, esto en realidad hace _más difícil_ que otros depuren el problema, ya que la información que está más arriba en el traceback también puede ser muy útil. Por lo tanto, una buena práctica es copiar y pegar el traceback _completo_, asegurándose de que esté bien formateado. Como estos tracebacks pueden volverse bastante largos, algunas personas prefieren mostrarlos después de haber explicado el código fuente. Hagamos esto. Ahora, nuestro tema del foro se ve de la siguiente manera:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic04.png" alt="Nuestro tema de ejemplo del foro, con el traceback completo." width="100%"/>
</div>

Esto es mucho más informativo, y un lector cuidadoso podría señalar que el problema parece deberse a pasar una entrada larga debido a esta línea en el traceback:

> Token indices sequence length is longer than the specified maximum sequence length for this model (583 > 512).

Sin embargo, podemos hacer las cosas aún más fáciles para ellos proporcionando el código real que provocó el error. Hagamos eso ahora.

### Proporcionar un ejemplo reproducible[[providing-a-reproducible-example]]

Si alguna vez has intentado depurar el código de otra persona, probablemente primero intentaste recrear el problema que reportaron para poder comenzar a trabajar a través del traceback para identificar el error. No es diferente cuando se trata de obtener (o dar) asistencia en los foros, así que realmente ayuda si puedes proporcionar un pequeño ejemplo que reproduzca el error. La mitad de las veces, simplemente realizar este ejercicio te ayudará a descubrir qué está mal. En cualquier caso, la pieza que falta en nuestro ejemplo es mostrar las _entradas_ que proporcionamos al modelo. Haciendo eso obtenemos algo como el siguiente ejemplo completado:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic05.png" alt="La versión final de nuestro tema del foro." width="100%"/>
</div>

Este tema ahora contiene bastante información, y está escrito de una manera que es mucho más probable que atraiga la atención de la comunidad y obtenga una respuesta útil. Con estas directrices básicas, ahora puedes crear excelentes temas para encontrar las respuestas a tus preguntas sobre Transformers.



---



# Depuracion del pipeline de entrenamiento[[debugging-the-training-pipeline]]


Has escrito un hermoso script para entrenar o ajustar finamente un modelo en una tarea determinada, siguiendo diligentemente los consejos del [Capitulo 7](/course/chapter7). Pero cuando lanzas el comando `trainer.train()`, algo horrible sucede: obtienes un error! O peor aun, todo parece estar bien y el entrenamiento se ejecuta sin errores, pero el modelo resultante es pesimo. En esta seccion, te mostraremos que puedes hacer para depurar este tipo de problemas.

## Depuracion del pipeline de entrenamiento[[debugging-the-training-pipeline]]


**Video:** [Ver en YouTube](https://youtu.be/L-WSwUWde1U)


El problema cuando encuentras un error en `trainer.train()` es que podria provenir de multiples fuentes, ya que el `Trainer` generalmente une muchas cosas. Convierte los datasets en dataloaders, por lo que el problema podria ser algo incorrecto en tu dataset, o algun problema al intentar agrupar elementos de los datasets. Luego toma un lote de datos y lo alimenta al modelo, por lo que el problema podria estar en el codigo del modelo. Despues de eso, calcula los gradientes y realiza el paso de optimizacion, por lo que el problema tambien podria estar en tu optimizador. E incluso si todo va bien durante el entrenamiento, algo podria salir mal durante la evaluacion si hay un problema con tu metrica.

La mejor manera de depurar un error que surge en `trainer.train()` es recorrer manualmente todo este pipeline para ver donde las cosas salieron mal. El error entonces suele ser muy facil de resolver.

Para demostrar esto, usaremos el siguiente script que (intenta) ajustar finamente un modelo DistilBERT en el [dataset MNLI](https://huggingface.co/datasets/glue):

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)
trainer.train()
```

Si intentas ejecutarlo, te encontraras con un error bastante criptico:

```python out
'ValueError: You have to specify either input_ids or inputs_embeds'
```

### Verifica tus datos[[check-your-data]]

Esto es obvio, pero si tus datos estan corruptos, el `Trainer` no podra formar lotes, y mucho menos entrenar tu modelo. Asi que lo primero es lo primero, necesitas echar un vistazo a lo que hay dentro de tu conjunto de entrenamiento.

Para evitar incontables horas tratando de arreglar algo que no es la fuente del error, te recomendamos que uses `trainer.train_dataset` para tus verificaciones y nada mas. Asi que hagamos eso aqui:

```py
trainer.train_dataset[0]
```

```python out
{'hypothesis': 'Product and geography are what make cream skimming work. ',
 'idx': 0,
 'label': 1,
 'premise': 'Conceptually cream skimming has two basic dimensions - product and geography.'}
```

Notas algo mal? Esto, junto con el mensaje de error sobre `input_ids` faltantes, deberia hacerte dar cuenta de que esos son textos, no numeros que el modelo pueda entender. Aqui, el error original es muy enganoso porque el `Trainer` elimina automaticamente las columnas que no coinciden con la firma del modelo (es decir, los argumentos esperados por el modelo). Eso significa que aqui, todo excepto las etiquetas fue descartado. Por lo tanto, no hubo problema con crear lotes y luego enviarlos al modelo, que a su vez se quejo de que no recibio la entrada adecuada.

Por que no se procesaron los datos? Usamos el metodo `Dataset.map()` en los datasets para aplicar el tokenizador en cada muestra. Pero si miras de cerca el codigo, veras que cometimos un error al pasar los conjuntos de entrenamiento y evaluacion al `Trainer`. En lugar de usar `tokenized_datasets` aqui, usamos `raw_datasets`. Asi que arreglemos esto!

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)
trainer.train()
```

Este nuevo codigo ahora dara un error diferente (progreso!):

```python out
'ValueError: expected sequence of length 43 at dim 1 (got 37)'
```

Mirando el traceback, podemos ver que el error ocurre en el paso de agrupacion de datos:

```python out
~/git/transformers/src/transformers/data/data_collator.py in torch_default_data_collator(features)
    105                 batch[k] = torch.stack([f[k] for f in features])
    106             else:
--> 107                 batch[k] = torch.tensor([f[k] for f in features])
    108
    109     return batch
```

Entonces, debemos pasar a eso. Sin embargo, antes de hacerlo, terminemos de inspeccionar nuestros datos, solo para estar 100% seguros de que son correctos.

Una cosa que siempre debes hacer cuando depuras una sesion de entrenamiento es echar un vistazo a las entradas decodificadas de tu modelo. No podemos entender los numeros que le damos directamente, asi que debemos mirar lo que esos numeros representan. En vision por computadora, por ejemplo, eso significa mirar las imagenes decodificadas de los pixeles que pasas, en voz significa escuchar las muestras de audio decodificadas, y para nuestro ejemplo de NLP aqui significa usar nuestro tokenizador para decodificar las entradas:

```py
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
```

```python out
'[CLS] conceptually cream skimming has two basic dimensions - product and geography. [SEP] product and geography are what make cream skimming work. [SEP]'
```

Eso parece correcto. Deberias hacer esto para todas las claves en las entradas:

```py
trainer.train_dataset[0].keys()
```

```python out
dict_keys(['attention_mask', 'hypothesis', 'idx', 'input_ids', 'label', 'premise'])
```

Ten en cuenta que las claves que no corresponden a entradas aceptadas por el modelo seran descartadas automaticamente, asi que aqui solo mantendremos `input_ids`, `attention_mask` y `label` (que sera renombrada a `labels`). Para verificar la firma del modelo, puedes imprimir la clase de tu modelo y luego ir a revisar su documentacion:

```py
type(trainer.model)
```

```python out
transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification
```

Entonces, en nuestro caso, podemos verificar los parametros aceptados en [esta pagina](https://huggingface.co/transformers/model_doc/distilbert.html#distilbertforsequenceclassification). El `Trainer` tambien registrara las columnas que esta descartando.

Hemos verificado que los IDs de entrada son correctos al decodificarlos. A continuacion esta la `attention_mask`:

```py
trainer.train_dataset[0]["attention_mask"]
```

```python out
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Como no aplicamos padding en nuestro preprocesamiento, esto parece perfectamente natural. Para asegurarnos de que no hay problema con esa mascara de atencion, verifiquemos que tiene la misma longitud que nuestros IDs de entrada:

```py
len(trainer.train_dataset[0]["attention_mask"]) == len(
    trainer.train_dataset[0]["input_ids"]
)
```

```python out
True
```

Eso esta bien! Por ultimo, verifiquemos nuestra etiqueta:

```py
trainer.train_dataset[0]["label"]
```

```python out
1
```

Como los IDs de entrada, este es un numero que no tiene mucho sentido por si solo. Como vimos antes, el mapeo entre enteros y nombres de etiquetas se almacena dentro del atributo `names` de la *caracteristica* correspondiente del dataset:

```py
trainer.train_dataset.features["label"].names
```

```python out
['entailment', 'neutral', 'contradiction']
```

Entonces `1` significa `neutral`, lo que significa que las dos oraciones que vimos arriba no estan en contradiccion, y la primera no implica la segunda. Eso parece correcto!

No tenemos IDs de tipo de token aqui, ya que DistilBERT no los espera; si tienes algunos en tu modelo, tambien deberias asegurarte de que coincidan correctamente con donde estan la primera y segunda oraciones en la entrada.

> [!TIP]
> Tu turno! Verifica que todo parece correcto con el segundo elemento del conjunto de entrenamiento.

Aqui solo estamos haciendo la verificacion en el conjunto de entrenamiento, pero por supuesto deberias verificar los conjuntos de validacion y prueba de la misma manera.

Ahora que sabemos que nuestros datasets se ven bien, es hora de verificar el siguiente paso del pipeline de entrenamiento.

### De datasets a dataloaders[[from-datasets-to-dataloaders]]

Lo siguiente que puede salir mal en el pipeline de entrenamiento es cuando el `Trainer` intenta formar lotes del conjunto de entrenamiento o validacion. Una vez que estes seguro de que los datasets del `Trainer` son correctos, puedes intentar formar un lote manualmente ejecutando lo siguiente (reemplaza `train` con `eval` para el dataloader de validacion):

```py
for batch in trainer.get_train_dataloader():
    break
```

Este codigo crea el dataloader de entrenamiento, luego itera a traves de el, deteniendose en la primera iteracion. Si el codigo se ejecuta sin error, tienes el primer lote de entrenamiento que puedes inspeccionar, y si el codigo produce un error, sabes con seguridad que el problema esta en el dataloader, como es el caso aqui:

```python out
~/git/transformers/src/transformers/data/data_collator.py in torch_default_data_collator(features)
    105                 batch[k] = torch.stack([f[k] for f in features])
    106             else:
--> 107                 batch[k] = torch.tensor([f[k] for f in features])
    108
    109     return batch

ValueError: expected sequence of length 45 at dim 1 (got 76)
```

Inspeccionar el ultimo frame del traceback deberia ser suficiente para darte una pista, pero hagamos un poco mas de investigacion. La mayoria de los problemas durante la creacion de lotes surgen debido a la agrupacion de ejemplos en un solo lote, asi que lo primero que debes verificar en caso de duda es que `collate_fn` esta usando tu `DataLoader`:

```py
data_collator = trainer.get_train_dataloader().collate_fn
data_collator
```

```python out
<function transformers.data.data_collator.default_data_collator(features: List[InputDataClass], return_tensors='pt') -> Dict[str, Any]>
```

Entonces esto es el `default_data_collator`, pero eso no es lo que queremos en este caso. Queremos rellenar nuestros ejemplos a la oracion mas larga en el lote, lo cual se hace mediante el collator `DataCollatorWithPadding`. Y se supone que este collator de datos es usado por defecto por el `Trainer`, entonces por que no se esta usando aqui?

La respuesta es que no pasamos el `tokenizer` al `Trainer`, por lo que no pudo crear el `DataCollatorWithPadding` que queremos. En la practica, nunca debes dudar en pasar explicitamente el collator de datos que quieres usar, para asegurarte de evitar este tipo de errores. Adaptemos nuestro codigo para hacer exactamente eso:

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

La buena noticia? Ya no obtenemos el mismo error que antes, lo cual es definitivamente progreso. La mala noticia? Obtenemos un infame error de CUDA en su lugar:

```python out
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

Esto es malo porque los errores de CUDA son extremadamente dificiles de depurar en general. Veremos en un momento como resolver esto, pero primero terminemos nuestro analisis de la creacion de lotes.

Si estas seguro de que tu collator de datos es el correcto, deberias intentar aplicarlo en un par de muestras de tu dataset:

```py
data_collator = trainer.get_train_dataloader().collate_fn
batch = data_collator([trainer.train_dataset[i] for i in range(4)])
```

Este codigo fallara porque el `train_dataset` contiene columnas de texto, que el `Trainer` normalmente elimina. Puedes eliminarlas manualmente, o si quieres replicar exactamente lo que el `Trainer` esta haciendo detras de escenas, puedes llamar al metodo privado `Trainer._remove_unused_columns()` que hace eso:

```py
data_collator = trainer.get_train_dataloader().collate_fn
actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)
batch = data_collator([actual_train_set[i] for i in range(4)])
```

Entonces deberias poder depurar manualmente lo que sucede dentro del collator de datos si el error persiste.

Ahora que hemos depurado el proceso de creacion de lotes, es hora de pasar uno a traves del modelo!

### Pasando a traves del modelo[[going-through-the-model]]

Deberias poder obtener un lote ejecutando el siguiente comando:

```py
for batch in trainer.get_train_dataloader():
    break
```

Si estas ejecutando este codigo en un notebook, podrias obtener un error de CUDA similar al que vimos antes, en cuyo caso necesitas reiniciar tu notebook y volver a ejecutar el ultimo fragmento sin la linea `trainer.train()`. Esa es la segunda cosa mas molesta de los errores de CUDA: rompen irremediablemente tu kernel. La cosa mas molesta de ellos es el hecho de que son dificiles de depurar.

Por que es eso? Tiene que ver con la forma en que funcionan las GPUs. Son extremadamente eficientes ejecutando muchas operaciones en paralelo, pero la desventaja es que cuando una de esas instrucciones resulta en un error, no lo sabes instantaneamente. Es solo cuando el programa llama a una sincronizacion de los multiples procesos en la GPU que se dara cuenta de que algo salio mal, por lo que el error se genera en un lugar que no tiene nada que ver con lo que lo creo. Por ejemplo, si miramos nuestro traceback anterior, el error se genero durante el pase hacia atras, pero veremos en un momento que en realidad proviene de algo en el pase hacia adelante.

Entonces, como depuramos esos errores? La respuesta es facil: no lo hacemos. A menos que tu error de CUDA sea un error de falta de memoria (lo que significa que no hay suficiente memoria en tu GPU), siempre debes volver a la CPU para depurarlo.

Para hacer esto en nuestro caso, solo tenemos que poner el modelo de vuelta en la CPU y llamarlo en nuestro lote -- el lote devuelto por el `DataLoader` aun no se ha movido a la GPU:

```python
outputs = trainer.model.cpu()(**batch)
```

```python out
~/.pyenv/versions/3.7.9/envs/base/lib/python3.7/site-packages/torch/nn/functional.py in nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)
   2386         )
   2387     if dim == 2:
-> 2388         ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
   2389     elif dim == 4:
   2390         ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

IndexError: Target 2 is out of bounds.
```

Entonces, la imagen se esta aclarando. En lugar de tener un error de CUDA, ahora tenemos un `IndexError` en el calculo de la perdida (asi que nada que ver con el pase hacia atras, como dijimos antes). Mas precisamente, podemos ver que es el objetivo 2 el que crea el error, asi que este es un muy buen momento para verificar el numero de etiquetas de nuestro modelo:

```python
trainer.model.config.num_labels
```

```python out
2
```

Con dos etiquetas, solo se permiten 0s y 1s como objetivos, pero segun el mensaje de error que obtuvimos, hay un 2. Obtener un 2 es en realidad normal: si recordamos los nombres de etiquetas que extrajimos antes, habia tres, asi que tenemos indices 0, 1 y 2 en nuestro dataset. El problema es que no le dijimos eso a nuestro modelo, que deberia haber sido creado con tres etiquetas. Asi que arreglemos eso!

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Aun no estamos incluyendo la linea `trainer.train()`, para tomarnos el tiempo de verificar que todo se ve bien. Si solicitamos un lote y lo pasamos a nuestro modelo, ahora funciona sin error!

```py
for batch in trainer.get_train_dataloader():
    break

outputs = trainer.model.cpu()(**batch)
```

El siguiente paso es entonces volver a la GPU y verificar que todo sigue funcionando:

```py
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: v.to(device) for k, v in batch.items()}

outputs = trainer.model.to(device)(**batch)
```

Si aun obtienes un error, asegurate de reiniciar tu notebook y solo ejecutar la ultima version del script.

### Realizando un paso de optimizacion[[performing-one-optimization-step]]

Ahora que sabemos que podemos construir lotes que realmente pasan a traves del modelo, estamos listos para el siguiente paso del pipeline de entrenamiento: calcular los gradientes y realizar un paso de optimizacion.

La primera parte es solo cuestion de llamar al metodo `backward()` en la perdida:

```py
loss = outputs.loss
loss.backward()
```

Es bastante raro obtener un error en esta etapa, pero si lo obtienes, asegurate de volver a la CPU para obtener un mensaje de error util.

Para realizar el paso de optimizacion, solo necesitamos crear el `optimizer` y llamar a su metodo `step()`:

```py
trainer.create_optimizer()
trainer.optimizer.step()
```

De nuevo, si estas usando el optimizador predeterminado en el `Trainer`, no deberias obtener un error en esta etapa, pero si tienes un optimizador personalizado, podria haber algunos problemas para depurar aqui. No olvides volver a la CPU si obtienes un error de CUDA extrano en esta etapa. Hablando de errores de CUDA, antes mencionamos un caso especial. Echemos un vistazo a eso ahora.

### Tratando con errores de CUDA por falta de memoria[[dealing-with-cuda-out-of-memory-errors]]

Cada vez que obtienes un mensaje de error que comienza con `RuntimeError: CUDA out of memory`, esto indica que te has quedado sin memoria de GPU. Esto no esta directamente relacionado con tu codigo, y puede suceder con un script que funciona perfectamente bien. Este error significa que intentaste poner demasiadas cosas en la memoria interna de tu GPU, y eso resulto en un error. Como con otros errores de CUDA, necesitaras reiniciar tu kernel para estar en un punto donde puedas ejecutar tu entrenamiento de nuevo.

Para resolver este problema, solo necesitas usar menos espacio de GPU -- algo que a menudo es mas facil decirlo que hacerlo. Primero, asegurate de no tener dos modelos en la GPU al mismo tiempo (a menos que eso sea necesario para tu problema, por supuesto). Luego, probablemente deberias reducir el tamano de tu lote, ya que afecta directamente los tamanos de todas las salidas intermedias del modelo y sus gradientes. Si el problema persiste, considera usar una version mas pequena de tu modelo.

> [!TIP]
> En la siguiente parte del curso, veremos tecnicas mas avanzadas que pueden ayudarte a reducir tu huella de memoria y permitirte ajustar finamente los modelos mas grandes.

### Evaluando el modelo[[evaluating-the-model]]

Ahora que hemos resuelto todos los problemas con nuestro codigo, todo es perfecto y el entrenamiento deberia ejecutarse sin problemas, verdad? No tan rapido! Si ejecutas el comando `trainer.train()`, todo se vera bien al principio, pero despues de un tiempo obtendras lo siguiente:

```py
# Esto tomara mucho tiempo y dara error, asi que no deberias ejecutar esta celda
trainer.train()
```

```python out
TypeError: only size-1 arrays can be converted to Python scalars
```

Te daras cuenta de que este error aparece durante la fase de evaluacion, asi que esto es lo ultimo que necesitaremos depurar.

Puedes ejecutar el bucle de evaluacion del `Trainer` independientemente del entrenamiento asi:

```py
trainer.evaluate()
```

```python out
TypeError: only size-1 arrays can be converted to Python scalars
```

> [!TIP]
> Siempre debes asegurarte de poder ejecutar `trainer.evaluate()` antes de lanzar `trainer.train()`, para evitar desperdiciar muchos recursos de computo antes de encontrar un error.

Antes de intentar depurar un problema en el bucle de evaluacion, primero debes asegurarte de haber revisado los datos, de poder formar un lote correctamente y de poder ejecutar tu modelo en el. Hemos completado todos esos pasos, asi que el siguiente codigo se puede ejecutar sin error:

```py
for batch in trainer.get_eval_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    outputs = trainer.model(**batch)
```

El error viene despues, al final de la fase de evaluacion, y si miramos el traceback vemos esto:

```python trace
~/git/datasets/src/datasets/metric.py in add_batch(self, predictions, references)
    431         """
    432         batch = {"predictions": predictions, "references": references}
--> 433         batch = self.info.features.encode_batch(batch)
    434         if self.writer is None:
    435             self._init_writer()
```

Esto nos dice que el error se origina en el modulo `datasets/metric.py` -- asi que este es un problema con nuestra funcion `compute_metrics()`. Toma una tupla con los logits y las etiquetas como arrays de NumPy, asi que intentemos alimentarla con eso:

```py
predictions = outputs.logits.cpu().numpy()
labels = batch["labels"].cpu().numpy()

compute_metrics((predictions, labels))
```

```python out
TypeError: only size-1 arrays can be converted to Python scalars
```

Obtenemos el mismo error, asi que el problema definitivamente esta en esa funcion. Si miramos de nuevo su codigo, vemos que solo esta reenviando las `predictions` y las `labels` a `metric.compute()`. Entonces, hay un problema con ese metodo? No realmente. Echemos un vistazo rapido a las formas:

```py
predictions.shape, labels.shape
```

```python out
((8, 3), (8,))
```

Nuestras predicciones todavia son logits, no las predicciones reales, por lo que la metrica esta devolviendo este error (algo oscuro). La solucion es bastante facil; solo tenemos que agregar un argmax en la funcion `compute_metrics()`:

```py
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


compute_metrics((predictions, labels))
```

```python out
{'accuracy': 0.625}
```

Ahora nuestro error esta arreglado! Este fue el ultimo, asi que nuestro script ahora entrenara un modelo correctamente.

Como referencia, aqui esta el script completamente arreglado:

```py
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

En este caso, no hay mas problemas, y nuestro script ajustara finamente un modelo que deberia dar resultados razonables. Pero, que podemos hacer cuando el entrenamiento procede sin ningun error, y el modelo entrenado no funciona bien en absoluto? Esa es la parte mas dificil del aprendizaje automatico, y te mostraremos algunas tecnicas que pueden ayudar.

> [!TIP]
> Si estas usando un bucle de entrenamiento manual, los mismos pasos se aplican para depurar tu pipeline de entrenamiento, pero es mas facil separarlos. Sin embargo, asegurate de no haber olvidado el `model.eval()` o `model.train()` en los lugares correctos, o el `zero_grad()` en cada paso!

## Depuracion de errores silenciosos durante el entrenamiento[[debugging-silent-errors-during-training]]

Que podemos hacer para depurar un entrenamiento que se completa sin error pero no obtiene buenos resultados? Te daremos algunos consejos aqui, pero ten en cuenta que este tipo de depuracion es la parte mas dificil del aprendizaje automatico, y no hay una respuesta magica.

### Verifica tus datos (de nuevo!)[[check-your-data-again]]

Tu modelo solo aprendera algo si realmente es posible aprender algo de tus datos. Si hay un error que corrompe los datos o las etiquetas se asignan aleatoriamente, es muy probable que no obtengas ningun modelo entrenado en tu dataset. Asi que siempre empieza verificando dos veces tus entradas y etiquetas decodificadas, y hazte las siguientes preguntas:

- Son comprensibles los datos decodificados?
- Estas de acuerdo con las etiquetas?
- Hay una etiqueta que es mas comun que las otras?
- Cual deberia ser la perdida/metrica si el modelo predijera una respuesta aleatoria/siempre la misma respuesta?

> [!WARNING]
> Si estas haciendo entrenamiento distribuido, imprime muestras de tu dataset en cada proceso y verifica tres veces que obtienes lo mismo. Un error comun es tener alguna fuente de aleatoriedad en la creacion de datos que hace que cada proceso tenga una version diferente del dataset.

Despues de mirar tus datos, revisa algunas de las predicciones del modelo y decodificalas tambien. Si el modelo siempre esta prediciendo lo mismo, podria ser porque tu dataset esta sesgado hacia una categoria (para problemas de clasificacion); tecnicas como el sobremuestreo de clases raras podrian ayudar.

Si la perdida/metrica que obtienes en tu modelo inicial es muy diferente de la perdida/metrica que esperarias para predicciones aleatorias, verifica dos veces la forma en que se calcula tu perdida o metrica, ya que probablemente hay un error ahi. Si estas usando varias perdidas que sumas al final, asegurate de que sean de la misma escala.

Cuando estes seguro de que tus datos son perfectos, puedes ver si el modelo es capaz de entrenar en ellos con una prueba simple.

### Sobreajusta tu modelo en un lote[[overfit-your-model-on-one-batch]]

El sobreajuste es algo que generalmente tratamos de evitar cuando entrenamos, ya que significa que el modelo no esta aprendiendo a reconocer las caracteristicas generales que queremos, sino que esta memorizando las muestras de entrenamiento. Sin embargo, intentar entrenar tu modelo en un lote una y otra vez es una buena prueba para verificar si el problema como lo planteaste puede ser resuelto por el modelo que estas intentando entrenar. Tambien te ayudara a ver si tu tasa de aprendizaje inicial es demasiado alta.

Hacer esto una vez que has definido tu `Trainer` es realmente facil; solo toma un lote de datos de entrenamiento, luego ejecuta un pequeno bucle de entrenamiento manual usando solo ese lote durante algo como 20 pasos:

```py
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}
trainer.create_optimizer()

for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
```

> [!TIP]
> Si tus datos de entrenamiento estan desbalanceados, asegurate de construir un lote de datos de entrenamiento que contenga todas las etiquetas.

El modelo resultante deberia tener resultados casi perfectos en el mismo `batch`. Calculemos la metrica en las predicciones resultantes:

```py
with torch.no_grad():
    outputs = trainer.model(**batch)
preds = outputs.logits
labels = batch["labels"]

compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))
```

```python out
{'accuracy': 1.0}
```

100% de precision, ahora este es un buen ejemplo de sobreajuste (lo que significa que si pruebas tu modelo en cualquier otra oracion, muy probablemente te dara una respuesta incorrecta)!

Si no logras que tu modelo obtenga resultados perfectos como este, significa que hay algo mal con la forma en que planteaste el problema o tus datos, asi que deberias arreglar eso. Solo cuando logres pasar la prueba de sobreajuste puedes estar seguro de que tu modelo realmente puede aprender algo.

> [!WARNING]
> Tendras que recrear tu modelo y tu `Trainer` despues de esta prueba, ya que el modelo obtenido probablemente no podra recuperarse y aprender algo util en tu dataset completo.

### No ajustes nada hasta que tengas una primera linea base[[dont-tune-anything-until-you-have-a-first-baseline]]

El ajuste de hiperparametros siempre se enfatiza como la parte mas dificil del aprendizaje automatico, pero es solo el ultimo paso para ayudarte a ganar un poco en la metrica. La mayoria de las veces, los hiperparametros predeterminados del `Trainer` funcionaran bien para darte buenos resultados, asi que no lances una busqueda de hiperparametros costosa y que consume tiempo hasta que tengas algo que supere la linea base que tienes en tu dataset.

Una vez que tengas un modelo suficientemente bueno, puedes empezar a ajustar un poco. No intentes lanzar mil ejecuciones con diferentes hiperparametros, sino compara un par de ejecuciones con diferentes valores para un hiperparametro para tener una idea de cual tiene el mayor impacto.

Si estas ajustando el modelo en si, mantenlo simple y no intentes nada que no puedas justificar razonablemente. Siempre asegurate de volver a la prueba de sobreajuste para verificar que tu cambio no ha tenido consecuencias no deseadas.

### Pide ayuda[[ask-for-help]]

Con suerte habras encontrado algun consejo en esta seccion que te ayudo a resolver tu problema, pero si ese no es el caso, recuerda que siempre puedes preguntar a la comunidad en los [foros](https://discuss.huggingface.co/).

Aqui hay algunos recursos adicionales que pueden ser utiles:

- ["Reproducibility as a vehicle for engineering best practices"](https://docs.google.com/presentation/d/1yHLPvPhUs2KGI5ZWo0sU-PKU3GimAk3iTsI38Z-B5Gw/edit#slide=id.p) por Joel Grus
- ["Checklist for debugging neural networks"](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) por Cecelia Shao
- ["How to unit test machine learning code"](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) por Chase Roberts
- ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) por Andrej Karpathy

Por supuesto, no todos los problemas que encuentres al entrenar redes neuronales son tu culpa! Si encuentras algo en la biblioteca de Transformers o Datasets que no parece correcto, es posible que hayas encontrado un error. Definitivamente deberias contarnoslo, y en la siguiente seccion explicaremos exactamente como hacerlo.


---



# Depuracion del pipeline de entrenamiento[[debugging-the-training-pipeline]]


Has escrito un script hermoso para entrenar o ajustar finamente un modelo en una tarea dada, siguiendo diligentemente los consejos del [Capitulo 7](/course/chapter7). Pero cuando ejecutas el comando `model.fit()`, algo horrible sucede: obtienes un error! O peor aun, todo parece estar bien y el entrenamiento se ejecuta sin errores, pero el modelo resultante es pesimo. En esta seccion, te mostraremos que puedes hacer para depurar este tipo de problemas.

## Depuracion del pipeline de entrenamiento[[debugging-the-training-pipeline]]


**Video:** [Ver en YouTube](https://youtu.be/N9kO52itd0Q)


El problema cuando encuentras un error en `model.fit()` es que podria venir de multiples fuentes, ya que el entrenamiento generalmente reune muchas cosas en las que has estado trabajando hasta ese momento. El problema podria ser algo incorrecto en tu conjunto de datos, o algun problema al intentar agrupar elementos de los conjuntos de datos. O podria ser algo incorrecto en el codigo del modelo, tu funcion de perdida u optimizador. E incluso si todo va bien durante el entrenamiento, algo podria salir mal durante la evaluacion si hay un problema con tu metrica.

La mejor manera de depurar un error que surge en `model.fit()` es recorrer manualmente todo este pipeline para ver donde las cosas salieron mal. El error entonces suele ser muy facil de resolver.

Para demostrar esto, usaremos el siguiente script que (intenta) ajustar finamente un modelo DistilBERT en el [conjunto de datos MNLI](https://huggingface.co/datasets/glue):

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "labels"], batch_size=16, shuffle=True
)

validation_dataset = tokenized_datasets["validation_matched"].to_tf_dataset(
    columns=["input_ids", "labels"], batch_size=16, shuffle=True
)

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

model.fit(train_dataset)
```

Si intentas ejecutarlo, podrias obtener algunos `VisibleDeprecationWarning` al hacer la conversion del conjunto de datos -- este es un problema conocido de experiencia de usuario que tenemos, asi que por favor ignoralo. Si estas leyendo el curso despues de, digamos, noviembre de 2021 y todavia esta sucediendo, entonces envia tweets enojados a @carrigmat hasta que lo arregle.

Sin embargo, un problema mas serio es que obtenemos un error directo. Y es realmente, aterradoramente largo:

```python out
ValueError: No gradients provided for any variable: ['tf_distil_bert_for_sequence_classification/distilbert/embeddings/word_embeddings/weight:0', '...']
```

Que significa eso? Intentamos entrenar con nuestros datos, pero no obtuvimos ningun gradiente? Esto es bastante desconcertante; como empezamos siquiera a depurar algo asi? Cuando el error que obtienes no sugiere inmediatamente donde esta el problema, la mejor solucion suele ser recorrer las cosas en secuencia, asegurandote en cada etapa de que todo se ve bien. Y por supuesto, el lugar para empezar siempre es...

### Revisa tus datos[[check-your-data]]

Esto es obvio, pero si tus datos estan corrompidos, Keras no va a poder arreglarlo por ti. Asi que lo primero es lo primero, necesitas echar un vistazo a lo que hay dentro de tu conjunto de entrenamiento.

Aunque es tentador mirar dentro de `raw_datasets` y `tokenized_datasets`, te recomendamos encarecidamente que vayas a los datos justo en el punto donde van a entrar al modelo. Eso significa leer una salida del `tf.data.Dataset` que creaste con la funcion `to_tf_dataset()`! Entonces, como hacemos eso? Los objetos `tf.data.Dataset` nos dan lotes completos a la vez y no soportan indexacion, asi que no podemos simplemente pedir `train_dataset[0]`. Sin embargo, podemos pedirle educadamente un lote:

```py
for batch in train_dataset:
    break
```

`break` termina el bucle despues de una iteracion, asi que esto toma el primer lote que sale de `train_dataset` y lo guarda como `batch`. Ahora, echemos un vistazo a lo que hay dentro:

```python out
{'attention_mask': <tf.Tensor: shape=(16, 76), dtype=int64, numpy=
 array([[1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]])>,
 'label': <tf.Tensor: shape=(16,), dtype=int64, numpy=array([0, 2, 1, 2, 1, 1, 2, 0, 0, 0, 1, 0, 1, 2, 2, 1])>,
 'input_ids': <tf.Tensor: shape=(16, 76), dtype=int64, numpy=
 array([[ 101, 2174, 1010, ...,    0,    0,    0],
        [ 101, 3174, 2420, ...,    0,    0,    0],
        [ 101, 2044, 2048, ...,    0,    0,    0],
        ...,
        [ 101, 3398, 3398, ..., 2051, 2894,  102],
        [ 101, 1996, 4124, ...,    0,    0,    0],
        [ 101, 1999, 2070, ...,    0,    0,    0]])>}
```

Esto se ve bien, no? Estamos pasando las `labels`, `attention_mask` e `input_ids` al modelo, que deberia ser todo lo que necesita para calcular las salidas y calcular la perdida. Entonces, por que no tenemos un gradiente? Mira mas de cerca: estamos pasando un solo diccionario como entrada, pero un lote de entrenamiento generalmente es un tensor de entrada o diccionario, mas un tensor de etiquetas. Nuestras etiquetas son solo una clave en nuestro diccionario de entrada.

Es esto un problema? No siempre, en realidad! Pero es uno de los problemas mas comunes que encontraras al entrenar modelos Transformer con TensorFlow. Todos nuestros modelos pueden calcular la perdida internamente, pero para hacer eso las etiquetas necesitan pasarse en el diccionario de entrada. Esta es la perdida que se usa cuando no especificamos un valor de perdida a `compile()`. Keras, por otro lado, generalmente espera que las etiquetas se pasen separadamente del diccionario de entrada, y los calculos de perdida generalmente fallaran si no haces eso.

El problema ahora se ha vuelto mas claro: pasamos un argumento `loss`, lo que significa que estamos pidiendo a Keras que calcule las perdidas por nosotros, pero pasamos nuestras etiquetas como entradas al modelo, no como etiquetas en el lugar donde Keras las espera! Necesitamos elegir una u otra: o usamos la perdida interna del modelo y mantenemos las etiquetas donde estan, o seguimos usando las perdidas de Keras, pero movemos las etiquetas al lugar donde Keras las espera. Por simplicidad, tomemos el primer enfoque. Cambia la llamada a `compile()` para que diga:

```py
model.compile(optimizer="adam")
```

Ahora usaremos la perdida interna del modelo, y este problema deberia resolverse!

> [!TIP]
> Tu turno! Como un desafio opcional despues de que hayamos resuelto los otros problemas, puedes intentar volver a este paso y hacer que el modelo funcione con la perdida original calculada por Keras en lugar de la perdida interna. Necesitaras agregar `"labels"` al argumento `label_cols` de `to_tf_dataset()` para asegurar que las etiquetas se emitan correctamente, lo que te dara gradientes -- pero hay un problema mas con la perdida que especificamos. El entrenamiento seguira ejecutandose con este problema, pero el aprendizaje sera muy lento y se estancara en una perdida de entrenamiento alta. Puedes descubrir cual es?
>
> Una pista codificada en ROT13, si estas atascado: Vs lbh ybbx ng gur bhgchgf bs FrdhraprPynffvsvpngvba zbqryf va Genafsbezref, gurve svefg bhgchg vf `ybtvgf`. Jung ner ybtvgf?
>
> Y una segunda pista: Jura lbh fcrpvsl bcgvzvmref, npgvingvbaf be ybffrf jvgu fgevatf, Xrenf frgf nyy gur nethzrag inyhrf gb gurve qrsnhygf. Jung nethzragf qbrf FcnefrPngrtbevpnyPebffragebcl unir, naq jung ner gurve qrsnhygf?

Ahora, intentemos entrenar. Deberiamos obtener gradientes ahora, asi que con suerte (musica ominosa suena aqui) podemos simplemente llamar a `model.fit()` y todo funcionara bien!

```python out
  246/24543 [..............................] - ETA: 15:52 - loss: nan
```

Oh no.

`nan` no es un valor de perdida muy alentador. Aun asi, hemos revisado nuestros datos, y se ven bastante bien. Si ese no es el problema, a donde podemos ir despues? El siguiente paso obvio es...

### Revisa tu modelo[[check-your-model]]

`model.fit()` es una funcion de conveniencia realmente genial en Keras, pero hace muchas cosas por ti, y eso puede hacer mas dificil encontrar exactamente donde ha ocurrido un problema. Si estas depurando tu modelo, una estrategia que realmente puede ayudar es pasar solo un lote al modelo, y mirar las salidas para ese lote en detalle. Otro consejo realmente util si el modelo esta lanzando errores es hacer `compile()` del modelo con `run_eagerly=True`. Esto lo hara mucho mas lento, pero hara los mensajes de error mucho mas comprensibles, porque indicaran exactamente donde en el codigo de tu modelo ocurrio el problema.

Por ahora, sin embargo, no necesitamos `run_eagerly` todavia. Ejecutemos el `batch` que obtuvimos antes a traves del modelo y veamos como se ven las salidas:

```py
model(batch)
```

```python out
TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(16,), dtype=float32, numpy=
array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan], dtype=float32)>, logits=<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
array([[nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan]], dtype=float32)>, hidden_states=None, attentions=None)
```

Bueno, esto es complicado. Todo es `nan`! Pero eso es extrano, no? Como se volverian todos nuestros logits `nan`? `nan` significa "no es un numero". Los valores `nan` ocurren frecuentemente cuando realizas una operacion prohibida, como division por cero. Pero algo que es muy importante saber sobre `nan` en aprendizaje automatico es que este valor tiende a *propagarse*. Si multiplicas un numero por `nan`, la salida tambien es `nan`. Y si obtienes un `nan` en cualquier lugar de tu salida, tu perdida o tu gradiente, entonces se propagara rapidamente a traves de todo tu modelo -- porque cuando ese valor `nan` se propaga hacia atras a traves de tu red, obtendras gradientes `nan`, y cuando las actualizaciones de pesos se calculan con esos gradientes, obtendras pesos `nan`, y esos pesos calcularan aun mas salidas `nan`! Pronto toda la red sera solo un gran bloque de `nan`s. Una vez que eso sucede, es bastante dificil ver donde comenzo el problema. Como podemos aislar donde `nan` aparecio por primera vez?

La respuesta es intentar *reinicializar* nuestro modelo. Una vez que comenzamos el entrenamiento, obtuvimos un `nan` en algun lugar y se propago rapidamente a traves de todo el modelo. Entonces, carguemos el modelo desde un checkpoint y no hagamos ninguna actualizacion de pesos, y veamos donde obtenemos un valor `nan`:

```py
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model(batch)
```

Cuando ejecutamos eso, obtenemos:

```py out
TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(16,), dtype=float32, numpy=
array([0.6844486 ,        nan,        nan, 0.67127866, 0.7068601 ,
              nan, 0.69309855,        nan, 0.65531296,        nan,
              nan,        nan, 0.675402  ,        nan,        nan,
       0.69831556], dtype=float32)>, logits=<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
array([[-0.04761693, -0.06509043],
       [-0.0481936 , -0.04556257],
       [-0.0040929 , -0.05848458],
       [-0.02417453, -0.0684005 ],
       [-0.02517801, -0.05241832],
       [-0.04514256, -0.0757378 ],
       [-0.02656011, -0.02646275],
       [ 0.00766164, -0.04350497],
       [ 0.02060014, -0.05655622],
       [-0.02615328, -0.0447021 ],
       [-0.05119278, -0.06928903],
       [-0.02859691, -0.04879177],
       [-0.02210129, -0.05791225],
       [-0.02363213, -0.05962167],
       [-0.05352269, -0.0481673 ],
       [-0.08141848, -0.07110836]], dtype=float32)>, hidden_states=None, attentions=None)
```

*Ahora* estamos llegando a algun lugar! No hay valores `nan` en nuestros logits, lo cual es tranquilizador. Pero vemos algunos valores `nan` en nuestra perdida! Hay algo en esas muestras en particular que esta causando este problema? Veamos cuales son (nota que si ejecutas este codigo tu mismo, podrias obtener indices diferentes porque el conjunto de datos ha sido mezclado):

```python
import numpy as np

loss = model(batch).loss.numpy()
indices = np.flatnonzero(np.isnan(loss))
indices
```

```python out
array([ 1,  2,  5,  7,  9, 10, 11, 13, 14])
```

Veamos las muestras de las que provienen estos indices:

```python
input_ids = batch["input_ids"].numpy()
input_ids[indices]
```

```python out
array([[  101,  2007,  2032,  2001,  1037, 16480,  3917,  2594,  4135,
        23212,  3070,  2214, 10170,  1010,  2012,  4356,  1997,  3183,
         6838, 12953,  2039,  2000,  1996,  6147,  1997,  2010,  2606,
         1012,   102,  6838,  2001,  3294,  6625,  3773,  1996,  2214,
         2158,  1012,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1998,  6814,  2016,  2234,  2461,  2153,  1998, 13322,
         2009,  1012,   102,  2045,  1005,  1055,  2053,  3382,  2008,
         2016,  1005,  2222,  3046,  8103,  2075,  2009,  2153,  1012,
          102,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1998,  2007,  1996,  3712,  4634,  1010,  2057,  8108,
         2025,  3404,  2028,  1012,  1996,  2616, 18449,  2125,  1999,
         1037,  9666,  1997,  4100,  8663, 11020,  6313,  2791,  1998,
         2431,  1011,  4301,  1012,   102,  2028,  1005,  1055,  5177,
         2110,  1998,  3977,  2000,  2832,  2106,  2025,  2689,  2104,
         2122,  6214,  1012,   102,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1045,  2001,  1999,  1037, 13090,  5948,  2007,  2048,
         2308,  2006,  2026,  5001,  2043,  2026,  2171,  2001,  2170,
         1012,   102,  1045,  2001,  3564,  1999,  2277,  1012,   102,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2195,  4279,  2191,  2039,  1996,  2181,  2124,  2004,
         1996,  2225,  7363,  1012,   102,  2045,  2003,  2069,  2028,
         2451,  1999,  1996,  2225,  7363,  1012,   102,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2061,  2008,  1045,  2123,  1005,  1056,  2113,  2065,
         2009,  2428, 10654,  7347,  2030,  2009,  7126,  2256,  2495,
         2291,   102,  2009,  2003,  5094,  2256,  2495,  2291,  2035,
         2105,  1012,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2051,  1010,  2029,  3216,  2019,  2503,  3444,  1010,
         6732,  1996,  2265,  2038, 19840,  2098,  2125,  9906,  1998,
         2003,  2770,  2041,  1997,  4784,  1012,   102,  2051,  6732,
         1996,  2265,  2003,  9525,  1998,  4569,  1012,   102,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1996, 10556,  2140, 11515,  2058,  1010,  2010,  2162,
         2252,  5689,  2013,  2010,  7223,  1012,   102,  2043,  1996,
        10556,  2140, 11515,  2058,  1010,  2010,  2252,  3062,  2000,
         1996,  2598,  1012,   102,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101, 13543,  1999,  2049,  6143,  2933,  2443,   102,  2025,
        13543,  1999,  6143,  2933,  2003,  2443,   102,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0]])
```

Bueno, hay mucho aqui, pero nada destaca como inusual. Veamos las etiquetas:

```python out
labels = batch['labels'].numpy()
labels[indices]
```

```python out
array([2, 2, 2, 2, 2, 2, 2, 2, 2])
```

Ah! Las muestras con `nan` todas tienen la misma etiqueta, y es la etiqueta 2. Esta es una pista muy fuerte. El hecho de que solo obtengamos una perdida de `nan` cuando nuestra etiqueta es 2 sugiere que este es un muy buen momento para verificar el numero de etiquetas en nuestro modelo:

```python
model.config.num_labels
```

```python out
2
```

Ahora vemos el problema: el modelo piensa que solo hay dos clases, pero las etiquetas llegan hasta 2, lo que significa que en realidad hay tres clases (porque 0 tambien es una clase). Asi es como obtuvimos un `nan` -- al intentar calcular la perdida para una clase inexistente! Intentemos cambiar eso y ajustar el modelo de nuevo:

```
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
model.compile(optimizer='adam')
model.fit(train_dataset)
```

```python out
  869/24543 [>.............................] - ETA: 15:29 - loss: 1.1032
```

Estamos entrenando! No mas `nan`s, y nuestra perdida esta disminuyendo... mas o menos. Si lo observas por un rato, podrias empezar a impacientarte un poco, porque el valor de la perdida se mantiene obstinadamente alto. Detengamos el entrenamiento aqui e intentemos pensar que podria estar causando este problema. En este punto, estamos bastante seguros de que tanto los datos como el modelo estan bien, pero nuestro modelo no esta aprendiendo bien. Que mas queda? Es hora de...

### Revisa tus hiperparametros[[check-your-hyperparameters]]

Si miras el codigo de arriba, podrias no ser capaz de ver ningun hiperparametro en absoluto, excepto quizas el `batch_size`, y ese no parece un culpable probable. Sin embargo, no te dejes enganar; siempre hay hiperparametros, y si no los puedes ver, solo significa que no sabes a que estan configurados. En particular, recuerda algo critico sobre Keras: si configuras una funcion de perdida, optimizador o activacion con una cadena de texto, _todos sus argumentos se configuraran a sus valores por defecto_. Esto significa que aunque usar cadenas para esto es muy conveniente, debes ser muy cuidadoso al hacerlo, ya que puede facilmente ocultarte cosas criticas. (Cualquiera que intente el desafio opcional de arriba deberia tomar nota cuidadosa de este hecho.)

En este caso, donde hemos configurado un argumento con una cadena? Estabamos configurando la perdida con una cadena inicialmente, pero ya no lo estamos haciendo. Sin embargo, estamos configurando el optimizador con una cadena. Podria eso estar ocultandonos algo? Echemos un vistazo a [sus argumentos](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).

Algo destaca aqui? Asi es -- la tasa de aprendizaje! Cuando solo usamos la cadena `'adam'`, vamos a obtener la tasa de aprendizaje por defecto, que es 0.001, o 1e-3. Esto es demasiado alto para un modelo Transformer! En general, recomendamos probar tasas de aprendizaje entre 1e-5 y 1e-4 para tus modelos; eso es algo entre 10X y 100X mas pequeno que el valor que realmente estamos usando aqui. Eso suena como que podria ser un problema importante, asi que intentemos reducirlo. Para hacer eso, necesitamos importar el objeto `optimizer` real. Mientras estamos en ello, reinicialicemos el modelo desde el checkpoint, en caso de que el entrenamiento con la alta tasa de aprendizaje haya danado sus pesos:

```python
from tensorflow.keras.optimizers import Adam

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.compile(optimizer=Adam(5e-5))
```

> [!TIP]
> Tambien puedes importar la funcion `create_optimizer()` de Transformers, que te dara un optimizador AdamW con decay de pesos correcto asi como warmup y decay de la tasa de aprendizaje. Este optimizador a menudo producira resultados ligeramente mejores que los que obtienes con el optimizador Adam por defecto.

Ahora, podemos intentar ajustar el modelo con la nueva y mejorada tasa de aprendizaje:

```python
model.fit(train_dataset)
```

```python out
319/24543 [..............................] - ETA: 16:07 - loss: 0.9718
```

Ahora nuestra perdida realmente esta yendo a algun lugar! El entrenamiento finalmente parece estar funcionando. Hay una leccion aqui: cuando tu modelo esta ejecutandose pero la perdida no esta disminuyendo, y estas seguro de que tus datos estan bien, es una buena idea revisar los hiperparametros como la tasa de aprendizaje y el decay de pesos. Configurar cualquiera de estos demasiado alto es muy probable que cause que el entrenamiento se "estanque" en un valor de perdida alto.

## Otros problemas potenciales[[other-potential-issues]]

Hemos cubierto los problemas en el script de arriba, pero hay varios otros errores comunes que podrias enfrentar. Echemos un vistazo a una lista (muy incompleta).

### Lidiar con errores de memoria insuficiente[[dealing-with-out-of-memory-errors]]

La senal reveladora de quedarse sin memoria es un error como "OOM when allocating tensor" -- OOM es la abreviatura de "out of memory" (sin memoria). Este es un peligro muy comun cuando se trabaja con modelos de lenguaje grandes. Si encuentras esto, una buena estrategia es reducir tu tamano de lote a la mitad e intentar de nuevo. Ten en cuenta, sin embargo, que algunos modelos son *muy* grandes. Por ejemplo, el GPT-2 de tamano completo tiene 1.5B parametros, lo que significa que necesitaras 6 GB de memoria solo para almacenar el modelo, y otros 6 GB para sus gradientes! Entrenar el modelo GPT-2 completo generalmente requerira mas de 20 GB de VRAM sin importar que tamano de lote uses, lo cual solo unas pocas GPUs tienen. Modelos mas ligeros como `distilbert-base-cased` son mucho mas faciles de ejecutar, y se entrenan mucho mas rapido tambien.

> [!TIP]
> En la siguiente parte del curso, veremos tecnicas mas avanzadas que pueden ayudarte a reducir tu huella de memoria y permitirte ajustar finamente los modelos mas grandes.

### TensorFlow Hambriento[[hungry-hungry-tensorflow]]

Una peculiaridad particular de TensorFlow de la que debes estar consciente es que asigna *toda* tu memoria de GPU para si mismo tan pronto como cargas un modelo o haces cualquier entrenamiento, y luego divide esa memoria segun sea necesario. Esto es diferente del comportamiento de otros frameworks, como PyTorch, que asignan memoria segun sea necesario con CUDA en lugar de hacerlo internamente. Una ventaja del enfoque de TensorFlow es que a menudo puede dar errores utiles cuando te quedas sin memoria, y puede recuperarse de ese estado sin bloquear todo el kernel de CUDA. Pero tambien hay una desventaja importante: si ejecutas dos procesos de TensorFlow al mismo tiempo, entonces **vas a tener un mal momento**.

Si estas ejecutando en Colab no necesitas preocuparte por esto, pero si estas ejecutando localmente esto es definitivamente algo con lo que debes tener cuidado. En particular, ten en cuenta que cerrar una pestana de notebook no necesariamente cierra ese notebook! Puede que necesites seleccionar los notebooks en ejecucion (los que tienen un icono verde) y cerrarlos manualmente en el listado del directorio. Cualquier notebook en ejecucion que estaba usando TensorFlow podria estar todavia reteniendo un monton de tu memoria de GPU, y eso significa que cualquier nuevo notebook que inicies podria encontrar algunos problemas muy extranos.

Si empiezas a obtener errores sobre CUDA, BLAS o cuBLAS en codigo que funcionaba antes, esto es muy a menudo el culpable. Puedes usar un comando como `nvidia-smi` para verificar -- cuando cierras o reinicias tu notebook actual, la mayor parte de tu memoria esta libre, o todavia esta en uso? Si todavia esta en uso, algo mas la esta reteniendo!

### Revisa tus datos (de nuevo!)[[check-your-data-again]]

Tu modelo solo aprendera algo si realmente es posible aprender algo de tus datos. Si hay un bug que corrompe los datos o las etiquetas se atribuyen aleatoriamente, es muy probable que no obtengas ningun entrenamiento del modelo en tu conjunto de datos. Una herramienta util aqui es `tokenizer.decode()`. Esto convertira los `input_ids` de vuelta en cadenas, para que puedas ver los datos y verificar si tus datos de entrenamiento estan ensenando lo que quieres que ensenen. Por ejemplo, despues de obtener un `batch` de tu `tf.data.Dataset` como hicimos arriba, puedes decodificar el primer elemento asi:

```py
input_ids = batch["input_ids"].numpy()
tokenizer.decode(input_ids[0])
```

Luego puedes compararlo con la primera etiqueta, asi:

```py
labels = batch["labels"].numpy()
label = labels[0]
```

Una vez que puedas ver tus datos de esta manera, puedes hacerte las siguientes preguntas:

- Son comprensibles los datos decodificados?
- Estas de acuerdo con las etiquetas?
- Hay una etiqueta que es mas comun que las otras?
- Cual deberia ser la perdida/metrica si el modelo predijera una respuesta aleatoria/siempre la misma respuesta?

Despues de mirar tus datos, revisa algunas de las predicciones del modelo -- si tu modelo produce tokens, intenta decodificarlos tambien! Si el modelo siempre predice lo mismo podria ser porque tu conjunto de datos esta sesgado hacia una categoria (para problemas de clasificacion), asi que tecnicas como sobremuestreo de clases raras podrian ayudar. Alternativamente, esto tambien puede ser causado por problemas de entrenamiento como malas configuraciones de hiperparametros.

Si la perdida/metrica que obtienes en tu modelo inicial antes de cualquier entrenamiento es muy diferente de la perdida/metrica que esperarias para predicciones aleatorias, verifica la forma en que se calcula tu perdida o metrica, ya que probablemente hay un bug ahi. Si estas usando varias perdidas que sumas al final, asegurate de que sean de la misma escala.

Cuando estes seguro de que tus datos son perfectos, puedes ver si el modelo es capaz de entrenar con ellos con una simple prueba.

### Sobreajusta tu modelo en un lote[[overfit-your-model-on-one-batch]]

El sobreajuste es usualmente algo que tratamos de evitar cuando entrenamos, ya que significa que el modelo no esta aprendiendo a reconocer las caracteristicas generales que queremos que reconozca sino que solo esta memorizando las muestras de entrenamiento. Sin embargo, intentar entrenar tu modelo en un lote una y otra vez es una buena prueba para verificar si el problema como lo planteaste puede ser resuelto por el modelo que estas intentando entrenar. Tambien te ayudara a ver si tu tasa de aprendizaje inicial es demasiado alta.

Hacer esto una vez que has definido tu `model` es realmente facil; solo toma un lote de datos de entrenamiento, luego trata ese `batch` como tu conjunto de datos completo, ajustandolo por un gran numero de epocas:

```py
for batch in train_dataset:
    break

# Asegurate de haber ejecutado model.compile() y configurado tu optimizador,
# y tu perdida/metricas si las estas usando

model.fit(batch, epochs=20)
```

> [!TIP]
> Si tus datos de entrenamiento estan desbalanceados, asegurate de construir un lote de datos de entrenamiento que contenga todas las etiquetas.

El modelo resultante deberia tener resultados casi perfectos en el `batch`, con una perdida que disminuye rapidamente hacia 0 (o el valor minimo para la perdida que estas usando).

Si no logras que tu modelo obtenga resultados perfectos como este, significa que hay algo mal con la forma en que planteaste el problema o con tus datos, asi que deberias arreglar eso. Solo cuando logres pasar la prueba de sobreajuste puedes estar seguro de que tu modelo realmente puede aprender algo.

> [!WARNING]
> Tendras que recrear tu modelo y recompilarlo despues de esta prueba de sobreajuste, ya que el modelo obtenido probablemente no podra recuperarse y aprender algo util en tu conjunto de datos completo.

### No ajustes nada hasta que tengas una primera linea base[[dont-tune-anything-until-you-have-a-first-baseline]]

El ajuste intenso de hiperparametros siempre se enfatiza como la parte mas dificil del aprendizaje automatico, pero es solo el ultimo paso para ayudarte a ganar un poco en la metrica. Valores *muy* malos para tus hiperparametros, como usar la tasa de aprendizaje por defecto de Adam de 1e-3 con un modelo Transformer, haran que el aprendizaje proceda muy lentamente o se detenga completamente, por supuesto, pero la mayoria de las veces hiperparametros "razonables", como una tasa de aprendizaje de 1e-5 a 5e-5, funcionaran bien para darte buenos resultados. Asi que, no te lances a una busqueda de hiperparametros que consume tiempo y es costosa hasta que tengas algo que supere la linea base que tienes en tu conjunto de datos.

Una vez que tengas un modelo suficientemente bueno, puedes empezar a ajustar un poco. No intentes lanzar mil ejecuciones con diferentes hiperparametros, sino compara un par de ejecuciones con diferentes valores para un hiperparametro para tener una idea de cual tiene el mayor impacto.

Si estas ajustando el modelo en si, mantenlo simple y no intentes nada que no puedas justificar razonablemente. Siempre asegurate de volver a la prueba de sobreajuste para verificar que tu cambio no haya tenido consecuencias no deseadas.

### Pide ayuda[[ask-for-help]]

Esperamos que hayas encontrado algun consejo en esta seccion que te haya ayudado a resolver tu problema, pero si ese no es el caso, recuerda que siempre puedes preguntar a la comunidad en los [foros](https://discuss.huggingface.co/).

Aqui hay algunos recursos adicionales que pueden resultar utiles:

- ["Reproducibility as a vehicle for engineering best practices"](https://docs.google.com/presentation/d/1yHLPvPhUs2KGI5ZWo0sU-PKU3GimAk3iTsI38Z-B5Gw/edit#slide=id.p) por Joel Grus
- ["Checklist for debugging neural networks"](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) por Cecelia Shao
- ["How to unit test machine learning code"](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) por Chase Roberts
- ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) por Andrej Karpathy

Por supuesto, no todos los problemas que encuentres al entrenar redes neuronales son tu culpa! Si encuentras algo en las bibliotecas de Transformers o Datasets que no parece correcto, puede que hayas encontrado un bug. Definitivamente deberias contarnoslo, y en la siguiente seccion te explicaremos exactamente como hacerlo.


---

# Cómo escribir un buen issue[[how-to-write-a-good-issue]]


Cuando encuentres algo que no parece estar bien con una de las bibliotecas de Hugging Face, definitivamente deberías hacérnoslo saber para que podamos arreglarlo (lo mismo aplica para cualquier biblioteca de código abierto, de hecho). Si no estás completamente seguro de si el bug está en tu propio código o en una de nuestras bibliotecas, el primer lugar para verificar son los [foros](https://discuss.huggingface.co/). La comunidad te ayudará a descubrirlo, y el equipo de Hugging Face también sigue de cerca las discusiones allí.


**Video:** [Ver en YouTube](https://youtu.be/_PAli-V4wj0)


Cuando estés seguro de que tienes un bug en tus manos, el primer paso es construir un ejemplo mínimo reproducible.

## Creando un ejemplo mínimo reproducible[[creating-a-minimal-reproducible-example]]

Es muy importante aislar la pieza de código que produce el bug, ya que nadie en el equipo de Hugging Face es un mago (todavía), y no pueden arreglar lo que no pueden ver. Un ejemplo mínimo reproducible debería, como el nombre indica, ser reproducible. Esto significa que no debería depender de ningún archivo externo o datos que puedas tener. Intenta reemplazar los datos que estás usando con algunos valores ficticios que se parezcan a los reales y que aún produzcan el mismo error.

> [!TIP]
> Muchos issues en el repositorio de Transformers quedan sin resolver porque los datos usados para reproducirlos no son accesibles.

Una vez que tengas algo que sea autocontenido, puedes intentar reducirlo a incluso menos líneas de código, construyendo lo que llamamos un _ejemplo mínimo reproducible_. Aunque esto requiere un poco más de trabajo de tu parte, casi tendrás garantizada la ayuda y una solución si proporcionas un buen y corto reproductor de bugs.

Si te sientes suficientemente cómodo, ve a inspeccionar el código fuente donde ocurre tu bug. Podrías encontrar una solución a tu problema (en cuyo caso incluso puedes sugerir un pull request para arreglarlo), pero más generalmente, esto puede ayudar a los mantenedores a entender mejor el código fuente cuando lean tu reporte.

## Completando la plantilla del issue[[filling-out-the-issue-template]]

Cuando registres tu issue, notarás que hay una plantilla para completar. Seguiremos la de [issues de Transformers](https://github.com/huggingface/transformers/issues/new/choose) aquí, pero el mismo tipo de información será requerida si reportas un issue en otro repositorio. No dejes la plantilla en blanco: tomarte el tiempo para completarla maximizará tus posibilidades de obtener una respuesta y resolver tu problema.

En general, al registrar un issue, siempre mantén la cortesía. Este es un proyecto de código abierto, así que estás usando software gratuito, y nadie tiene ninguna obligación de ayudarte. Puedes incluir lo que consideres críticas justificadas en tu issue, pero entonces los mantenedores podrían tomarlo mal y no tener prisa por ayudarte. Asegúrate de leer el [código de conducta](https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md) del proyecto.

### Incluyendo información de tu entorno[[including-your-environment-information]]

Transformers proporciona una utilidad para obtener toda la información que necesitamos sobre tu entorno. Simplemente escribe lo siguiente en tu terminal:

```
transformers-cli env
```

y deberías obtener algo como esto:

```out
Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.

- `transformers` version: 4.12.0.dev0
- Platform: Linux-5.10.61-1-MANJARO-x86_64-with-arch-Manjaro-Linux
- Python version: 3.7.9
- PyTorch version (GPU?): 1.8.1+cu111 (True)
- Tensorflow version (GPU?): 2.5.0 (True)
- Flax version (CPU?/GPU?/TPU?): 0.3.4 (cpu)
- Jax version: 0.2.13
- JaxLib version: 0.1.65
- Using GPU in script?: <fill in>
- Using distributed or parallel set-up in script?: <fill in>
```

También puedes agregar un `!` al principio del comando `transformers-cli env` para ejecutarlo desde una celda de notebook, y luego copiar y pegar el resultado al principio de tu issue.

### Etiquetando personas[[tagging-people]]

Etiquetar personas escribiendo un `@` seguido de su nombre de usuario de GitHub les enviará una notificación para que vean tu issue y puedan responder más rápido. Usa esto con moderación, porque las personas que etiquetes podrían no apreciar ser notificadas si es algo con lo que no tienen relación directa. Si has revisado los archivos fuente relacionados con tu bug, deberías etiquetar a la última persona que hizo cambios en la línea que crees que es responsable de tu problema (puedes encontrar esta información mirando dicha línea en GitHub, seleccionándola, y luego haciendo clic en "View git blame").

De lo contrario, la plantilla ofrece sugerencias de personas para etiquetar. En general, nunca etiquetes a más de tres personas.

### Incluyendo un ejemplo reproducible[[including-a-reproducible-example]]

Si has logrado crear un ejemplo autocontenido que produce el bug, ahora es el momento de incluirlo. Escribe una línea con tres acentos graves seguidos de `python`, así:

```
```python
```

luego pega tu ejemplo mínimo reproducible y escribe una nueva línea con tres acentos graves. Esto asegurará que tu código esté formateado correctamente.

Si no lograste crear un ejemplo reproducible, explica en pasos claros cómo llegaste a tu issue. Incluye un enlace a un notebook de Google Colab donde obtuviste el error si puedes. Cuanta más información compartas, mejor podrán los mantenedores responderte.

En todos los casos, deberías copiar y pegar el mensaje de error completo que estás recibiendo. Si estás trabajando en Colab, recuerda que algunos de los marcos pueden estar automáticamente colapsados en el stack trace, así que asegúrate de expandirlos antes de copiar. Como con el ejemplo de código, coloca ese mensaje de error entre dos líneas con tres acentos graves, para que esté formateado correctamente.

### Describiendo el comportamiento esperado[[describing-the-expected-behavior]]

Explica en unas pocas líneas lo que esperabas obtener, para que los mantenedores comprendan completamente el problema. Esta parte es generalmente bastante obvia, por lo que debería caber en una oración, pero en algunos casos podrías tener mucho que decir.

## Y luego qué?[[and-then-what]]

Una vez que tu issue esté registrado, asegúrate de verificar rápidamente que todo se vea bien. Puedes editar el issue si cometiste un error, o incluso cambiar su título si te das cuenta de que el problema es diferente de lo que inicialmente pensaste.

No tiene sentido hacer ping a las personas si no obtienes una respuesta. Si nadie te ayuda en unos días, es probable que nadie pudo entender tu problema. No dudes en volver al ejemplo reproducible. Puedes hacerlo más corto y más directo? Si no obtienes una respuesta en una semana, puedes dejar un mensaje pidiendo ayuda amablemente, especialmente si has editado tu issue para incluir más información sobre el problema.


---

# Parte 2 completada![[part-2-completed]]


¡Felicidades, has completado la segunda parte del curso! Estamos trabajando activamente en la tercera, así que suscríbete a nuestro [boletín](https://huggingface.curated.co/) para asegurarte de no perderte su lanzamiento.

Ahora deberías poder abordar una variedad de tareas de NLP, y hacer fine-tuning o preentrenar un modelo en ellas. ¡No olvides compartir tus resultados con la comunidad en el [Model Hub](https://huggingface.co/models)!

¡Estamos ansiosos por ver lo que construirás con el conocimiento que has adquirido!


---



# Cuestionario de fin de capítulo[[end-of-chapter-quiz]]


¡Pongamos a prueba lo que aprendiste en este capítulo!

### 1. ¿En qué orden debes leer un traceback de Python?


- De arriba hacia abajo
- De abajo hacia arriba


### 2. ¿Qué es un ejemplo mínimo reproducible?


- Una implementación simple de una arquitectura Transformer de un artículo de investigación
- Un bloque de código compacto y autocontenido que puede ejecutarse sin dependencias externas de archivos o datos privados
- Una captura de pantalla del traceback de Python
- Un notebook que contiene todo tu análisis, incluyendo partes no relacionadas con el error


### 3. Supongamos que intentas ejecutar el siguiente código, que lanza un error:

```py
from transformers import GPT3ForSequenceClassification

# ImportError: cannot import name 'GPT3ForSequenceClassification' from 'transformers' (/Users/lewtun/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/__init__.py)
# ---------------------------------------------------------------------------
# ImportError                               Traceback (most recent call last)
# /var/folders/28/k4cy5q7s2hs92xq7_h89_vgm0000gn/T/ipykernel_30848/333858878.py in <module>
# ----> 1 from transformers import GPT3ForSequenceClassification

# ImportError: cannot import name 'GPT3ForSequenceClassification' from 'transformers' (/Users/lewtun/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/__init__.py)
```

¿Cuál de las siguientes podría ser una buena opción para el título de un tema en el foro para pedir ayuda?


- <code>ImportError: cannot import name 
- Problema con `from transformers import GPT3ForSequenceClassification`
- ¿Por qué no puedo importar `GPT3ForSequenceClassification`?
- ¿Está GPT-3 soportado en 🤗 Transformers?


### 4. Supongamos que has intentado ejecutar `trainer.train()` y te enfrentas a un error críptico que no te dice exactamente de dónde viene el error. ¿Cuál de los siguientes es el primer lugar donde deberías buscar errores en tu pipeline de entrenamiento?


- El paso de optimización donde calculamos gradientes y realizamos la retropropagación
- El paso de evaluación donde calculamos métricas
- Los datasets
- Los dataloaders


### 5. ¿Cuál es la mejor manera de depurar un error de CUDA?


- Publicar el mensaje de error en los foros o GitHub.
- Ejecutar el mismo código en la CPU.
- Leer el traceback para averiguar qué causó el error.
- Reducir el tamaño del batch.
- Reiniciar el kernel de Jupyter.


### 6. ¿Cuál es la mejor manera de conseguir que se solucione un issue en GitHub?


- Publicar un ejemplo completo reproducible del bug.
- Pedir actualizaciones todos los días.
- Inspeccionar el código fuente alrededor del bug e intentar encontrar la razón por la que ocurre. Publicar los resultados en el issue.


### 7. ¿Por qué hacer overfitting a un batch suele ser una buena técnica de depuración?


- No lo es; el overfitting siempre es malo y debe evitarse.
- Nos permite verificar que el modelo es capaz de reducir la pérdida a cero.
- Nos permite verificar que las formas de los tensores de nuestras entradas y etiquetas son correctas.


### 8. ¿Por qué es una buena idea incluir detalles sobre tu entorno de cómputo con `transformers-cli env` cuando creas un nuevo issue en el repositorio de 🤗 Transformers?


- Permite a los mantenedores entender qué versión de la biblioteca estás usando.
- Permite a los mantenedores saber si estás ejecutando código en Windows, macOS o Linux.
- Permite a los mantenedores saber si estás ejecutando código en una GPU o CPU.



---
