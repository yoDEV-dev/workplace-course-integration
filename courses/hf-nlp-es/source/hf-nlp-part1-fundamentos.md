# Curso de NLP de Hugging Face 🤗
## Parte 1: Fundamentos

**Capítulos 0-4:** Configuración, Modelos de Transformadores, Usando Transformers, Ajuste fino, Compartiendo modelos

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

