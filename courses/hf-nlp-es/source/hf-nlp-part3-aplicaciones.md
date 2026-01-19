# Curso de NLP de Hugging Face 🤗
## Parte 3: Aplicaciones

**Capítulos 7-8:** Tareas clásicas de NLP, Cómo solicitar ayuda

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
