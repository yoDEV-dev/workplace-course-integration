# Curso de NLP de Hugging Face 🤗
## Parte 2: Bibliotecas

**Capítulos 5-6:** La librería Datasets, La librería Tokenizers

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

