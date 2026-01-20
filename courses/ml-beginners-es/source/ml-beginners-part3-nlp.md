# ML para Principiantes 🤖
## Parte 3: Procesamiento de Lenguaje Natural
**NLP y Análisis de Texto**

---


# Procesamiento de Lenguaje Natural

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1eb379dc2d0c9940b320732d16083778",
  "translation_date": "2025-09-04T00:33:20+00:00",
  "source_file": "6-NLP/README.md",
  "language_code": "es"
}
-->
# Comenzando con el procesamiento de lenguaje natural

El procesamiento de lenguaje natural (NLP, por sus siglas en inglés) es la capacidad de un programa de computadora para entender el lenguaje humano tal como se habla y se escribe, conocido como lenguaje natural. Es un componente de la inteligencia artificial (IA). El NLP ha existido por más de 50 años y tiene raíces en el campo de la lingüística. Todo el campo está dirigido a ayudar a las máquinas a entender y procesar el lenguaje humano. Esto puede ser utilizado para realizar tareas como la corrección ortográfica o la traducción automática. Tiene una variedad de aplicaciones en el mundo real en varios campos, incluyendo la investigación médica, los motores de búsqueda y la inteligencia empresarial.

## Tema regional: Idiomas y literatura europeas y hoteles románticos de Europa ❤️

En esta sección del programa, se te presentará uno de los usos más extendidos del aprendizaje automático: el procesamiento de lenguaje natural (NLP). Derivado de la lingüística computacional, esta categoría de inteligencia artificial es el puente entre los humanos y las máquinas a través de la comunicación por voz o texto.

En estas lecciones aprenderemos los conceptos básicos del NLP construyendo pequeños bots conversacionales para entender cómo el aprendizaje automático ayuda a que estas conversaciones sean cada vez más 'inteligentes'. Viajarás en el tiempo, conversando con Elizabeth Bennett y el Sr. Darcy del clásico de Jane Austen, **Orgullo y Prejuicio**, publicado en 1813. Luego, ampliarás tus conocimientos aprendiendo sobre el análisis de sentimientos a través de reseñas de hoteles en Europa.

![Libro de Orgullo y Prejuicio y té](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/p&p.279f1c49ecd88941.webp)
> Foto por <a href="https://unsplash.com/@elaineh?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Elaine Howlin</a> en <a href="https://unsplash.com/s/photos/pride-and-prejudice?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Lecciones

1. [Introducción al procesamiento de lenguaje natural](1-Introduction-to-NLP/README.md)
2. [Tareas y técnicas comunes de NLP](2-Tasks/README.md)
3. [Traducción y análisis de sentimientos con aprendizaje automático](3-Translation-Sentiment/README.md)
4. [Preparando tus datos](4-Hotel-Reviews-1/README.md)
5. [NLTK para análisis de sentimientos](5-Hotel-Reviews-2/README.md)

## Créditos 

Estas lecciones de procesamiento de lenguaje natural fueron escritas con ☕ por [Stephen Howell](https://twitter.com/Howell_MSFT)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-04T22:28:35+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "es"
}
-->
# Introducción al procesamiento de lenguaje natural

Esta lección cubre una breve historia y conceptos importantes del *procesamiento de lenguaje natural*, un subcampo de la *lingüística computacional*.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

El procesamiento de lenguaje natural, conocido como NLP por sus siglas en inglés, es una de las áreas más conocidas donde se ha aplicado el aprendizaje automático y se utiliza en software de producción.

✅ ¿Puedes pensar en algún software que uses todos los días que probablemente tenga algo de NLP integrado? ¿Qué hay de tus programas de procesamiento de texto o aplicaciones móviles que usas regularmente?

Aprenderás sobre:

- **La idea de los idiomas**. Cómo se desarrollaron los idiomas y cuáles han sido las principales áreas de estudio.
- **Definición y conceptos**. También aprenderás definiciones y conceptos sobre cómo las computadoras procesan texto, incluyendo análisis sintáctico, gramática e identificación de sustantivos y verbos. Hay algunas tareas de codificación en esta lección, y se introducen varios conceptos importantes que aprenderás a programar más adelante en las próximas lecciones.

## Lingüística computacional

La lingüística computacional es un área de investigación y desarrollo que, durante muchas décadas, ha estudiado cómo las computadoras pueden trabajar con los idiomas, e incluso entenderlos, traducirlos y comunicarse con ellos. El procesamiento de lenguaje natural (NLP) es un campo relacionado que se centra en cómo las computadoras pueden procesar idiomas 'naturales', es decir, humanos.

### Ejemplo - dictado en el teléfono

Si alguna vez has dictado a tu teléfono en lugar de escribir o le has hecho una pregunta a un asistente virtual, tu voz fue convertida en texto y luego procesada o *analizada* desde el idioma que hablaste. Las palabras clave detectadas se procesaron en un formato que el teléfono o asistente pudo entender y actuar en consecuencia.

![comprensión](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> ¡La comprensión lingüística real es difícil! Imagen por [Jen Looper](https://twitter.com/jenlooper)

### ¿Cómo es posible esta tecnología?

Esto es posible porque alguien escribió un programa de computadora para hacerlo. Hace unas décadas, algunos escritores de ciencia ficción predijeron que las personas hablarían principalmente con sus computadoras y que estas siempre entenderían exactamente lo que querían decir. Lamentablemente, resultó ser un problema más difícil de lo que muchos imaginaron, y aunque hoy en día es un problema mucho mejor entendido, existen desafíos significativos para lograr un procesamiento de lenguaje natural 'perfecto' cuando se trata de entender el significado de una oración. Este es un problema particularmente difícil cuando se trata de entender el humor o detectar emociones como el sarcasmo en una oración.

En este punto, podrías estar recordando las clases escolares donde el profesor cubría las partes de la gramática en una oración. En algunos países, se enseña gramática y lingüística como una materia dedicada, pero en muchos, estos temas se incluyen como parte del aprendizaje de un idioma: ya sea tu primer idioma en la escuela primaria (aprendiendo a leer y escribir) y quizás un segundo idioma en la escuela secundaria. ¡No te preocupes si no eres un experto en diferenciar sustantivos de verbos o adverbios de adjetivos!

Si tienes dificultades con la diferencia entre el *presente simple* y el *presente progresivo*, no estás solo. Esto es algo desafiante para muchas personas, incluso hablantes nativos de un idioma. La buena noticia es que las computadoras son muy buenas aplicando reglas formales, y aprenderás a escribir código que pueda *analizar* una oración tan bien como un humano. El mayor desafío que examinarás más adelante es entender el *significado* y el *sentimiento* de una oración.

## Prerrequisitos

Para esta lección, el principal prerrequisito es poder leer y entender el idioma de esta lección. No hay problemas matemáticos ni ecuaciones que resolver. Aunque el autor original escribió esta lección en inglés, también está traducida a otros idiomas, por lo que podrías estar leyendo una traducción. Hay ejemplos donde se utilizan varios idiomas diferentes (para comparar las diferentes reglas gramaticales de distintos idiomas). Estos *no* están traducidos, pero el texto explicativo sí lo está, por lo que el significado debería ser claro.

Para las tareas de codificación, usarás Python y los ejemplos están en Python 3.8.

En esta sección, necesitarás y usarás:

- **Comprensión de Python 3**. Comprensión del lenguaje de programación en Python 3, esta lección utiliza entrada, bucles, lectura de archivos, arreglos.
- **Visual Studio Code + extensión**. Usaremos Visual Studio Code y su extensión de Python. También puedes usar un IDE de Python de tu elección.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) es una biblioteca simplificada de procesamiento de texto para Python. Sigue las instrucciones en el sitio de TextBlob para instalarlo en tu sistema (instala también los corpora, como se muestra a continuación):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Consejo: Puedes ejecutar Python directamente en entornos de VS Code. Consulta los [documentos](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para más información.

## Hablando con máquinas

La historia de intentar que las computadoras entiendan el lenguaje humano se remonta a décadas atrás, y uno de los primeros científicos en considerar el procesamiento de lenguaje natural fue *Alan Turing*.

### La 'prueba de Turing'

Cuando Turing investigaba la *inteligencia artificial* en la década de 1950, consideró si se podría realizar una prueba conversacional entre un humano y una computadora (a través de correspondencia escrita) donde el humano en la conversación no estuviera seguro de si estaba conversando con otro humano o con una computadora.

Si, después de cierto tiempo de conversación, el humano no podía determinar si las respuestas provenían de una computadora o no, ¿podría decirse que la computadora estaba *pensando*?

### La inspiración - 'el juego de imitación'

La idea para esto provino de un juego de fiesta llamado *El juego de imitación*, donde un interrogador está solo en una habitación y tiene la tarea de determinar cuál de dos personas (en otra habitación) es hombre y cuál es mujer. El interrogador puede enviar notas y debe tratar de pensar en preguntas cuyas respuestas escritas revelen el género de la persona misteriosa. Por supuesto, los jugadores en la otra habitación intentan engañar al interrogador respondiendo preguntas de manera que lo confundan o lo engañen, mientras dan la apariencia de responder honestamente.

### Desarrollando Eliza

En la década de 1960, un científico del MIT llamado *Joseph Weizenbaum* desarrolló [*Eliza*](https://wikipedia.org/wiki/ELIZA), una 'terapeuta' computarizada que hacía preguntas al humano y daba la apariencia de entender sus respuestas. Sin embargo, aunque Eliza podía analizar una oración e identificar ciertos constructos gramaticales y palabras clave para dar una respuesta razonable, no podía decirse que *entendiera* la oración. Si a Eliza se le presentaba una oración con el formato "**Yo estoy** <u>triste</u>", podría reorganizar y sustituir palabras en la oración para formar la respuesta "¿Cuánto tiempo has **estado** <u>triste</u>?".

Esto daba la impresión de que Eliza entendía la declaración y estaba haciendo una pregunta de seguimiento, mientras que en realidad estaba cambiando el tiempo verbal y agregando algunas palabras. Si Eliza no podía identificar una palabra clave para la que tuviera una respuesta, en su lugar daba una respuesta aleatoria que podría aplicarse a muchas declaraciones diferentes. Eliza podía ser fácilmente engañada, por ejemplo, si un usuario escribía "**Tú eres** una <u>bicicleta</u>", podría responder con "¿Cuánto tiempo he **sido** una <u>bicicleta</u>?", en lugar de una respuesta más razonada.

[![Conversando con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversando con Eliza")

> 🎥 Haz clic en la imagen de arriba para ver un video sobre el programa original de ELIZA

> Nota: Puedes leer la descripción original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada en 1966 si tienes una cuenta de ACM. Alternativamente, lee sobre Eliza en [Wikipedia](https://wikipedia.org/wiki/ELIZA).

## Ejercicio - programando un bot conversacional básico

Un bot conversacional, como Eliza, es un programa que solicita la entrada del usuario y parece entender y responder de manera inteligente. A diferencia de Eliza, nuestro bot no tendrá varias reglas que le den la apariencia de tener una conversación inteligente. En cambio, nuestro bot tendrá una sola habilidad: mantener la conversación con respuestas aleatorias que podrían funcionar en casi cualquier conversación trivial.

### El plan

Tus pasos al construir un bot conversacional:

1. Imprimir instrucciones que aconsejen al usuario cómo interactuar con el bot.
2. Iniciar un bucle:
   1. Aceptar la entrada del usuario.
   2. Si el usuario ha pedido salir, entonces salir.
   3. Procesar la entrada del usuario y determinar la respuesta (en este caso, la respuesta es una elección aleatoria de una lista de posibles respuestas genéricas).
   4. Imprimir la respuesta.
3. Volver al paso 2.

### Construyendo el bot

Vamos a crear el bot a continuación. Comenzaremos definiendo algunas frases.

1. Crea este bot tú mismo en Python con las siguientes respuestas aleatorias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aquí hay un ejemplo de salida para guiarte (la entrada del usuario está en las líneas que comienzan con `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Una posible solución a la tarea está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py).

    ✅ Detente y reflexiona

    1. ¿Crees que las respuestas aleatorias podrían 'engañar' a alguien para que piense que el bot realmente las entendió?
    2. ¿Qué características necesitaría el bot para ser más efectivo?
    3. Si un bot realmente pudiera 'entender' el significado de una oración, ¿necesitaría 'recordar' el significado de oraciones anteriores en una conversación también?

---

## 🚀Desafío

Elige uno de los elementos de "detente y reflexiona" anteriores y trata de implementarlo en código o escribe una solución en papel usando pseudocódigo.

En la próxima lección, aprenderás sobre una serie de otros enfoques para analizar el lenguaje natural y el aprendizaje automático.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Echa un vistazo a las referencias a continuación como oportunidades de lectura adicional.

### Referencias

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## Tarea

[Busca un bot](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-04T22:27:00+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "es"
}
-->
# Tareas y técnicas comunes de procesamiento de lenguaje natural

Para la mayoría de las tareas de *procesamiento de lenguaje natural*, el texto que se va a procesar debe descomponerse, examinarse y los resultados almacenarse o cruzarse con reglas y conjuntos de datos. Estas tareas permiten al programador derivar el _significado_, la _intención_ o solo la _frecuencia_ de términos y palabras en un texto.

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

Descubramos técnicas comunes utilizadas en el procesamiento de texto. Combinadas con aprendizaje automático, estas técnicas te ayudan a analizar grandes cantidades de texto de manera eficiente. Sin embargo, antes de aplicar ML a estas tareas, entendamos los problemas que enfrenta un especialista en NLP.

## Tareas comunes en NLP

Existen diferentes formas de analizar un texto con el que estás trabajando. Hay tareas que puedes realizar y, a través de ellas, puedes comprender el texto y sacar conclusiones. Generalmente, estas tareas se llevan a cabo en una secuencia.

### Tokenización

Probablemente, lo primero que la mayoría de los algoritmos de NLP tienen que hacer es dividir el texto en tokens o palabras. Aunque esto suena simple, tener en cuenta la puntuación y los delimitadores de palabras y oraciones en diferentes idiomas puede complicarlo. Es posible que tengas que usar varios métodos para determinar las demarcaciones.

![tokenización](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/2-Tasks/images/tokenization.png)
> Tokenizando una oración de **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) son una forma de convertir tus datos de texto en valores numéricos. Los embeddings se realizan de manera que las palabras con un significado similar o palabras que se usan juntas se agrupen.

![word embeddings](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/2-Tasks/images/embedding.png)
> "Tengo el mayor respeto por tus nervios, son mis viejos amigos." - Word embeddings para una oración en **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

✅ Prueba [esta herramienta interesante](https://projector.tensorflow.org/) para experimentar con word embeddings. Al hacer clic en una palabra, se muestran grupos de palabras similares: 'juguete' se agrupa con 'disney', 'lego', 'playstation' y 'consola'.

### Parsing y etiquetado de partes del discurso

Cada palabra que ha sido tokenizada puede etiquetarse como una parte del discurso: un sustantivo, verbo o adjetivo. La oración `el rápido zorro rojo saltó sobre el perro marrón perezoso` podría etiquetarse como POS con zorro = sustantivo, saltó = verbo.

![parsing](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/2-Tasks/images/parse.png)

> Analizando una oración de **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

El parsing consiste en reconocer qué palabras están relacionadas entre sí en una oración; por ejemplo, `el rápido zorro rojo saltó` es una secuencia de adjetivo-sustantivo-verbo que está separada de la secuencia `el perro marrón perezoso`.

### Frecuencia de palabras y frases

Un procedimiento útil al analizar un gran cuerpo de texto es construir un diccionario de cada palabra o frase de interés y cuántas veces aparece. La frase `el rápido zorro rojo saltó sobre el perro marrón perezoso` tiene una frecuencia de palabras de 2 para "el".

Veamos un texto de ejemplo donde contamos la frecuencia de palabras. El poema Los Ganadores de Rudyard Kipling contiene el siguiente verso:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Como las frecuencias de frases pueden ser sensibles o insensibles a mayúsculas según se requiera, la frase `un amigo` tiene una frecuencia de 2, `el` tiene una frecuencia de 6 y `viajes` tiene una frecuencia de 2.

### N-grams

Un texto puede dividirse en secuencias de palabras de una longitud establecida: una sola palabra (unigram), dos palabras (bigramas), tres palabras (trigramas) o cualquier número de palabras (n-grams).

Por ejemplo, `el rápido zorro rojo saltó sobre el perro marrón perezoso` con un puntaje n-gram de 2 produce los siguientes n-grams:

1. el rápido  
2. rápido zorro  
3. zorro rojo  
4. rojo saltó  
5. saltó sobre  
6. sobre el  
7. el perro  
8. perro marrón  
9. marrón perezoso  

Podría ser más fácil visualizarlo como una caja deslizante sobre la oración. Aquí está para n-grams de 3 palabras, el n-gram está en negrita en cada oración:

1.   <u>**el rápido zorro**</u> saltó sobre el perro marrón perezoso  
2.   el **<u>rápido zorro rojo</u>** saltó sobre el perro marrón perezoso  
3.   el rápido **<u>zorro rojo saltó</u>** sobre el perro marrón perezoso  
4.   el rápido zorro **<u>rojo saltó sobre</u>** el perro marrón perezoso  
5.   el rápido zorro rojo **<u>saltó sobre el</u>** perro marrón perezoso  
6.   el rápido zorro rojo saltó **<u>sobre el perro</u>** marrón perezoso  
7.   el rápido zorro rojo saltó sobre <u>**el perro marrón**</u> perezoso  
8.   el rápido zorro rojo saltó sobre el **<u>perro marrón perezoso</u>**  

![ventana deslizante n-grams](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/2-Tasks/images/n-grams.gif)

> Valor n-gram de 3: Infografía por [Jen Looper](https://twitter.com/jenlooper)

### Extracción de frases nominales

En la mayoría de las oraciones, hay un sustantivo que es el sujeto u objeto de la oración. En inglés, a menudo se identifica porque tiene 'a', 'an' o 'the' antes de él. Identificar el sujeto u objeto de una oración mediante la 'extracción de la frase nominal' es una tarea común en NLP al intentar comprender el significado de una oración.

✅ En la oración "No puedo fijar la hora, ni el lugar, ni la mirada ni las palabras, que sentaron las bases. Hace demasiado tiempo. Estaba en medio antes de darme cuenta de que había comenzado.", ¿puedes identificar las frases nominales?

En la oración `el rápido zorro rojo saltó sobre el perro marrón perezoso` hay 2 frases nominales: **rápido zorro rojo** y **perro marrón perezoso**.

### Análisis de sentimiento

Una oración o texto puede analizarse para determinar el sentimiento, o cuán *positivo* o *negativo* es. El sentimiento se mide en *polaridad* y *objetividad/subjetividad*. La polaridad se mide de -1.0 a 1.0 (negativo a positivo) y de 0.0 a 1.0 (más objetivo a más subjetivo).

✅ Más adelante aprenderás que hay diferentes formas de determinar el sentimiento utilizando aprendizaje automático, pero una forma es tener una lista de palabras y frases que han sido categorizadas como positivas o negativas por un experto humano y aplicar ese modelo al texto para calcular un puntaje de polaridad. ¿Puedes ver cómo esto funcionaría en algunas circunstancias y menos en otras?

### Inflexión

La inflexión te permite tomar una palabra y obtener su forma singular o plural.

### Lematización

Un *lema* es la raíz o palabra principal de un conjunto de palabras; por ejemplo, *voló*, *vuela*, *volando* tienen como lema el verbo *volar*.

También hay bases de datos útiles disponibles para el investigador de NLP, en particular:

### WordNet

[WordNet](https://wordnet.princeton.edu/) es una base de datos de palabras, sinónimos, antónimos y muchos otros detalles para cada palabra en muchos idiomas diferentes. Es increíblemente útil al intentar construir traducciones, correctores ortográficos o herramientas de lenguaje de cualquier tipo.

## Bibliotecas de NLP

Afortunadamente, no tienes que construir todas estas técnicas tú mismo, ya que hay excelentes bibliotecas de Python disponibles que hacen que sea mucho más accesible para desarrolladores que no están especializados en procesamiento de lenguaje natural o aprendizaje automático. Las próximas lecciones incluyen más ejemplos de estas, pero aquí aprenderás algunos ejemplos útiles para ayudarte con la próxima tarea.

### Ejercicio - usando la biblioteca `TextBlob`

Usemos una biblioteca llamada TextBlob, ya que contiene APIs útiles para abordar este tipo de tareas. TextBlob "se basa en los hombros gigantes de [NLTK](https://nltk.org) y [pattern](https://github.com/clips/pattern), y funciona bien con ambos." Tiene una cantidad considerable de ML integrado en su API.

> Nota: Una útil [Guía rápida](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) está disponible para TextBlob y se recomienda para desarrolladores experimentados en Python.

Al intentar identificar *frases nominales*, TextBlob ofrece varias opciones de extractores para encontrarlas.

1. Echa un vistazo a `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > ¿Qué está pasando aquí? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) es "Un extractor de frases nominales que utiliza el análisis de fragmentos entrenado con el corpus de entrenamiento ConLL-2000." ConLL-2000 se refiere a la Conferencia de Aprendizaje Computacional de Lenguaje Natural de 2000. Cada año, la conferencia organizaba un taller para abordar un problema difícil de NLP, y en 2000 fue el análisis de fragmentos nominales. Se entrenó un modelo en el Wall Street Journal, con "las secciones 15-18 como datos de entrenamiento (211727 tokens) y la sección 20 como datos de prueba (47377 tokens)". Puedes ver los procedimientos utilizados [aquí](https://www.clips.uantwerpen.be/conll2000/chunking/) y los [resultados](https://ifarm.nl/erikt/research/np-chunking.html).

### Desafío - mejorando tu bot con NLP

En la lección anterior construiste un bot de preguntas y respuestas muy simple. Ahora, harás que Marvin sea un poco más empático analizando tu entrada para determinar el sentimiento y mostrando una respuesta que coincida con el sentimiento. También necesitarás identificar una `frase nominal` y preguntar sobre ella.

Tus pasos al construir un bot conversacional mejorado:

1. Imprime instrucciones aconsejando al usuario cómo interactuar con el bot.  
2. Inicia un bucle:  
   1. Acepta la entrada del usuario.  
   2. Si el usuario ha pedido salir, entonces sal.  
   3. Procesa la entrada del usuario y determina una respuesta de sentimiento adecuada.  
   4. Si se detecta una frase nominal en el sentimiento, pluralízala y pide más información sobre ese tema.  
   5. Imprime la respuesta.  
3. Regresa al paso 2.  

Aquí está el fragmento de código para determinar el sentimiento usando TextBlob. Nota que solo hay cuatro *gradientes* de respuesta de sentimiento (puedes tener más si lo deseas):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Aquí hay un ejemplo de salida para guiarte (la entrada del usuario está en las líneas que comienzan con >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Una posible solución a la tarea está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Verificación de conocimiento

1. ¿Crees que las respuestas empáticas podrían 'engañar' a alguien para que piense que el bot realmente los entiende?  
2. ¿Hace que el bot sea más 'creíble' identificar la frase nominal?  
3. ¿Por qué sería útil extraer una 'frase nominal' de una oración?  

---

Implementa el bot en la verificación de conocimiento anterior y pruébalo con un amigo. ¿Puede engañarlos? ¿Puedes hacer que tu bot sea más 'creíble'?

## 🚀Desafío

Toma una tarea de la verificación de conocimiento anterior e intenta implementarla. Prueba el bot con un amigo. ¿Puede engañarlos? ¿Puedes hacer que tu bot sea más 'creíble'?

## [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

En las próximas lecciones aprenderás más sobre análisis de sentimiento. Investiga esta técnica interesante en artículos como estos en [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tarea 

[Haz que un bot responda](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-04T22:29:03+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "es"
}
-->
# Traducción y análisis de sentimientos con ML

En las lecciones anteriores aprendiste cómo construir un bot básico utilizando `TextBlob`, una biblioteca que incorpora ML detrás de escena para realizar tareas básicas de PLN como la extracción de frases nominales. Otro desafío importante en la lingüística computacional es la _traducción_ precisa de una oración de un idioma hablado o escrito a otro.

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

La traducción es un problema muy difícil, agravado por el hecho de que hay miles de idiomas y cada uno puede tener reglas gramaticales muy diferentes. Una de las aproximaciones es convertir las reglas gramaticales formales de un idioma, como el inglés, en una estructura independiente del idioma, y luego traducirla convirtiéndola nuevamente a otro idioma. Este enfoque implica los siguientes pasos:

1. **Identificación**. Identificar o etiquetar las palabras en el idioma de entrada como sustantivos, verbos, etc.
2. **Crear la traducción**. Producir una traducción directa de cada palabra en el formato del idioma de destino.

### Ejemplo de oración, inglés a irlandés

En 'inglés', la oración _I feel happy_ tiene tres palabras en el orden:

- **sujeto** (I)
- **verbo** (feel)
- **adjetivo** (happy)

Sin embargo, en el idioma 'irlandés', la misma oración tiene una estructura gramatical muy diferente: las emociones como "*happy*" o "*sad*" se expresan como algo que está *sobre* ti.

La frase en inglés `I feel happy` en irlandés sería `Tá athas orm`. Una traducción *literal* sería `Happy is upon me`.

Un hablante de irlandés que traduce al inglés diría `I feel happy`, no `Happy is upon me`, porque entiende el significado de la oración, incluso si las palabras y la estructura de la oración son diferentes.

El orden formal de la oración en irlandés es:

- **verbo** (Tá o is)
- **adjetivo** (athas, o happy)
- **sujeto** (orm, o upon me)

## Traducción

Un programa de traducción ingenuo podría traducir solo palabras, ignorando la estructura de la oración.

✅ Si has aprendido un segundo (o tercer o más) idioma como adulto, es posible que hayas comenzado pensando en tu idioma nativo, traduciendo un concepto palabra por palabra en tu cabeza al segundo idioma y luego expresando tu traducción. Esto es similar a lo que hacen los programas de traducción computacional ingenuos. ¡Es importante superar esta fase para alcanzar la fluidez!

La traducción ingenua lleva a malas (y a veces hilarantes) malas traducciones: `I feel happy` se traduce literalmente como `Mise bhraitheann athas` en irlandés. Eso significa (literalmente) `me feel happy` y no es una oración válida en irlandés. Aunque el inglés y el irlandés son idiomas hablados en dos islas vecinas, son idiomas muy diferentes con estructuras gramaticales distintas.

> Puedes ver algunos videos sobre las tradiciones lingüísticas irlandesas como [este](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Enfoques de aprendizaje automático

Hasta ahora, has aprendido sobre el enfoque de reglas formales para el procesamiento del lenguaje natural. Otro enfoque es ignorar el significado de las palabras y _en su lugar usar aprendizaje automático para detectar patrones_. Esto puede funcionar en la traducción si tienes muchos textos (un *corpus*) o textos (*corpora*) en ambos idiomas, el de origen y el de destino.

Por ejemplo, considera el caso de *Orgullo y Prejuicio*, una novela inglesa muy conocida escrita por Jane Austen en 1813. Si consultas el libro en inglés y una traducción humana del libro en *francés*, podrías detectar frases en uno que se traducen _idiomáticamente_ al otro. Harás esto en un momento.

Por ejemplo, cuando una frase en inglés como `I have no money` se traduce literalmente al francés, podría convertirse en `Je n'ai pas de monnaie`. "Monnaie" es un falso cognado francés complicado, ya que 'money' y 'monnaie' no son sinónimos. Una mejor traducción que un humano podría hacer sería `Je n'ai pas d'argent`, porque transmite mejor el significado de que no tienes dinero (en lugar de 'cambio suelto', que es el significado de 'monnaie').

![monnaie](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Imagen por [Jen Looper](https://twitter.com/jenlooper)

Si un modelo de ML tiene suficientes traducciones humanas para construir un modelo, puede mejorar la precisión de las traducciones identificando patrones comunes en textos que han sido previamente traducidos por hablantes humanos expertos de ambos idiomas.

### Ejercicio - traducción

Puedes usar `TextBlob` para traducir oraciones. Prueba la famosa primera línea de **Orgullo y Prejuicio**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` hace un buen trabajo con la traducción: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Se puede argumentar que la traducción de TextBlob es mucho más precisa, de hecho, que la traducción francesa de 1932 del libro por V. Leconte y Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

En este caso, la traducción informada por ML hace un mejor trabajo que el traductor humano que pone innecesariamente palabras en la boca del autor original para 'claridad'.

> ¿Qué está pasando aquí? ¿Y por qué TextBlob es tan bueno en la traducción? Bueno, detrás de escena, está utilizando Google Translate, una IA sofisticada capaz de analizar millones de frases para predecir las mejores cadenas para la tarea en cuestión. No hay nada manual aquí y necesitas una conexión a internet para usar `blob.translate`.

✅ Prueba algunas oraciones más. ¿Cuál es mejor, la traducción por ML o la humana? ¿En qué casos?

## Análisis de sentimientos

Otra área donde el aprendizaje automático puede funcionar muy bien es el análisis de sentimientos. Un enfoque no basado en ML para el sentimiento es identificar palabras y frases que son 'positivas' y 'negativas'. Luego, dado un nuevo texto, calcular el valor total de las palabras positivas, negativas y neutrales para identificar el sentimiento general. 

Este enfoque es fácilmente engañado, como habrás visto en la tarea de Marvin: la oración `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` es una oración sarcástica de sentimiento negativo, pero el algoritmo simple detecta 'great', 'wonderful', 'glad' como positivas y 'waste', 'lost' y 'dark' como negativas. El sentimiento general se ve influido por estas palabras conflictivas.

✅ Detente un momento y piensa en cómo transmitimos sarcasmo como hablantes humanos. La inflexión del tono juega un papel importante. Intenta decir la frase "Well, that film was awesome" de diferentes maneras para descubrir cómo tu voz transmite significado.

### Enfoques de ML

El enfoque de ML sería reunir manualmente cuerpos de texto negativos y positivos: tweets, reseñas de películas o cualquier cosa donde el humano haya dado una puntuación *y* una opinión escrita. Luego se pueden aplicar técnicas de PLN a las opiniones y puntuaciones, para que emerjan patrones (por ejemplo, las reseñas de películas positivas tienden a tener la frase 'Oscar worthy' más que las reseñas negativas, o las reseñas positivas de restaurantes dicen 'gourmet' mucho más que 'disgusting').

> ⚖️ **Ejemplo**: Si trabajas en la oficina de un político y se está debatiendo una nueva ley, los ciudadanos podrían escribir correos electrónicos a la oficina apoyando o en contra de la ley en particular. Supongamos que te encargan leer los correos electrónicos y clasificarlos en 2 grupos, *a favor* y *en contra*. Si hubiera muchos correos electrónicos, podrías sentirte abrumado intentando leerlos todos. ¿No sería genial si un bot pudiera leerlos todos por ti, entenderlos y decirte en qué grupo pertenece cada correo electrónico? 
> 
> Una forma de lograr esto es usar aprendizaje automático. Entrenarías el modelo con una parte de los correos electrónicos *en contra* y una parte de los correos electrónicos *a favor*. El modelo tendería a asociar frases y palabras con el lado en contra y el lado a favor, *pero no entendería ninguno de los contenidos*, solo que ciertas palabras y patrones son más propensos a aparecer en un correo electrónico *en contra* o *a favor*. Podrías probarlo con algunos correos electrónicos que no hayas usado para entrenar el modelo y ver si llega a la misma conclusión que tú. Luego, una vez que estés satisfecho con la precisión del modelo, podrías procesar correos electrónicos futuros sin tener que leer cada uno.

✅ ¿Este proceso te suena similar a procesos que has usado en lecciones anteriores?

## Ejercicio - oraciones sentimentales

El sentimiento se mide con una *polaridad* de -1 a 1, donde -1 es el sentimiento más negativo y 1 es el más positivo. El sentimiento también se mide con una puntuación de 0 - 1 para objetividad (0) y subjetividad (1).

Echa otro vistazo a *Orgullo y Prejuicio* de Jane Austen. El texto está disponible aquí en [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). El siguiente ejemplo muestra un programa corto que analiza el sentimiento de las primeras y últimas oraciones del libro y muestra su polaridad de sentimiento y puntuación de subjetividad/objetividad.

Debes usar la biblioteca `TextBlob` (descrita anteriormente) para determinar el `sentimiento` (no necesitas escribir tu propio calculador de sentimientos) en la siguiente tarea.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Ves el siguiente resultado:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Desafío - verificar polaridad de sentimientos

Tu tarea es determinar, utilizando la polaridad de sentimientos, si *Orgullo y Prejuicio* tiene más oraciones absolutamente positivas que absolutamente negativas. Para esta tarea, puedes asumir que una puntuación de polaridad de 1 o -1 es absolutamente positiva o negativa respectivamente.

**Pasos:**

1. Descarga una [copia de Orgullo y Prejuicio](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) de Project Gutenberg como un archivo .txt. Elimina los metadatos al inicio y al final del archivo, dejando solo el texto original.
2. Abre el archivo en Python y extrae el contenido como una cadena.
3. Crea un TextBlob utilizando la cadena del libro.
4. Analiza cada oración del libro en un bucle.
   1. Si la polaridad es 1 o -1, almacena la oración en un array o lista de mensajes positivos o negativos.
5. Al final, imprime todas las oraciones positivas y negativas (por separado) y el número de cada una.

Aquí hay una [solución de ejemplo](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Verificación de conocimiento

1. El sentimiento se basa en las palabras utilizadas en la oración, pero ¿el código *entiende* las palabras?
2. ¿Crees que la polaridad de sentimientos es precisa, o en otras palabras, ¿estás de acuerdo con las puntuaciones?
   1. En particular, ¿estás de acuerdo o en desacuerdo con la polaridad absolutamente **positiva** de las siguientes oraciones?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Las siguientes 3 oraciones fueron puntuadas con un sentimiento absolutamente positivo, pero al leerlas detenidamente, no son oraciones positivas. ¿Por qué el análisis de sentimientos pensó que eran oraciones positivas?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. ¿Estás de acuerdo o en desacuerdo con la polaridad absolutamente **negativa** de las siguientes oraciones?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Cualquier aficionado a Jane Austen entenderá que ella a menudo usa sus libros para criticar los aspectos más ridículos de la sociedad de la Regencia inglesa. Elizabeth Bennett, el personaje principal en *Orgullo y Prejuicio*, es una observadora social aguda (como la autora) y su lenguaje a menudo está muy matizado. Incluso Mr. Darcy (el interés amoroso en la historia) nota el uso juguetón y burlón del lenguaje por parte de Elizabeth: "He tenido el placer de conocerte lo suficiente como para saber que disfrutas mucho ocasionalmente profesando opiniones que en realidad no son tuyas".

---

## 🚀Desafío

¿Puedes mejorar a Marvin aún más extrayendo otras características de la entrada del usuario?

## [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio
Hay muchas maneras de extraer el sentimiento de un texto. Piensa en las aplicaciones empresariales que podrían utilizar esta técnica. Reflexiona sobre cómo podría salir mal. Lee más sobre sistemas sofisticados y listos para empresas que analizan sentimientos, como [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Prueba algunas de las frases de Orgullo y Prejuicio mencionadas anteriormente y observa si puede detectar matices.

## Tarea

[Licencia poética](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T22:27:31+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "es"
}
-->
# Análisis de sentimientos con reseñas de hoteles - procesando los datos

En esta sección, utilizarás las técnicas de las lecciones anteriores para realizar un análisis exploratorio de datos en un conjunto de datos grande. Una vez que tengas una buena comprensión de la utilidad de las diferentes columnas, aprenderás:

- cómo eliminar las columnas innecesarias
- cómo calcular nuevos datos basados en las columnas existentes
- cómo guardar el conjunto de datos resultante para usarlo en el desafío final

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

### Introducción

Hasta ahora has aprendido que los datos de texto son bastante diferentes de los datos numéricos. Si el texto fue escrito o hablado por un humano, puede analizarse para encontrar patrones, frecuencias, sentimientos y significados. Esta lección te lleva a un conjunto de datos real con un desafío real: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, que incluye una [licencia CC0: Dominio Público](https://creativecommons.org/publicdomain/zero/1.0/). Fue recopilado de Booking.com a partir de fuentes públicas. El creador del conjunto de datos es Jiashen Liu.

### Preparación

Necesitarás:

* La capacidad de ejecutar notebooks .ipynb usando Python 3
* pandas
* NLTK, [que deberías instalar localmente](https://www.nltk.org/install.html)
* El conjunto de datos disponible en Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Tiene un tamaño aproximado de 230 MB descomprimido. Descárgalo en la carpeta raíz `/data` asociada con estas lecciones de NLP.

## Análisis exploratorio de datos

Este desafío asume que estás construyendo un bot de recomendación de hoteles utilizando análisis de sentimientos y puntuaciones de reseñas de huéspedes. El conjunto de datos que usarás incluye reseñas de 1493 hoteles diferentes en 6 ciudades.

Usando Python, un conjunto de datos de reseñas de hoteles y el análisis de sentimientos de NLTK, podrías descubrir:

* ¿Cuáles son las palabras y frases más utilizadas en las reseñas?
* ¿Las *etiquetas* oficiales que describen un hotel se correlacionan con las puntuaciones de las reseñas (por ejemplo, hay reseñas más negativas para un hotel en particular por *Familia con niños pequeños* que por *Viajero solo*, lo que podría indicar que es mejor para *Viajeros solos*)?
* ¿Las puntuaciones de sentimientos de NLTK "coinciden" con la puntuación numérica del revisor del hotel?

#### Conjunto de datos

Exploremos el conjunto de datos que has descargado y guardado localmente. Abre el archivo en un editor como VS Code o incluso Excel.

Los encabezados en el conjunto de datos son los siguientes:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aquí están agrupados de una manera que podría ser más fácil de examinar: 
##### Columnas del hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitud), `lng` (longitud)
  * Usando *lat* y *lng* podrías trazar un mapa con Python mostrando las ubicaciones de los hoteles (quizás codificado por colores para reseñas negativas y positivas)
  * Hotel_Address no parece ser útil para nosotros, y probablemente lo reemplazaremos con un país para facilitar la clasificación y búsqueda

**Columnas de meta-reseñas del hotel**

* `Average_Score`
  * Según el creador del conjunto de datos, esta columna es la *Puntuación promedio del hotel, calculada en base al último comentario del último año*. Esto parece una forma inusual de calcular la puntuación, pero es el dato recopilado, así que por ahora lo tomaremos como válido.
  
  ✅ Basándote en las otras columnas de este conjunto de datos, ¿puedes pensar en otra forma de calcular la puntuación promedio?

* `Total_Number_of_Reviews`
  * El número total de reseñas que ha recibido este hotel - no está claro (sin escribir algo de código) si esto se refiere a las reseñas en el conjunto de datos.
* `Additional_Number_of_Scoring`
  * Esto significa que se dio una puntuación de reseña pero no se escribió una reseña positiva o negativa por parte del revisor.

**Columnas de reseñas**

- `Reviewer_Score`
  - Este es un valor numérico con un máximo de 1 decimal entre los valores mínimos y máximos 2.5 y 10
  - No se explica por qué 2.5 es la puntuación más baja posible
- `Negative_Review`
  - Si un revisor no escribió nada, este campo tendrá "**No Negative**"
  - Ten en cuenta que un revisor puede escribir una reseña positiva en la columna de reseña negativa (por ejemplo, "no hay nada malo en este hotel")
- `Review_Total_Negative_Word_Counts`
  - Un mayor conteo de palabras negativas indica una puntuación más baja (sin verificar la sentimentalidad)
- `Positive_Review`
  - Si un revisor no escribió nada, este campo tendrá "**No Positive**"
  - Ten en cuenta que un revisor puede escribir una reseña negativa en la columna de reseña positiva (por ejemplo, "no hay nada bueno en este hotel en absoluto")
- `Review_Total_Positive_Word_Counts`
  - Un mayor conteo de palabras positivas indica una puntuación más alta (sin verificar la sentimentalidad)
- `Review_Date` y `days_since_review`
  - Se podría aplicar una medida de frescura o antigüedad a una reseña (las reseñas más antiguas podrían no ser tan precisas como las más recientes debido a cambios en la gestión del hotel, renovaciones, adición de una piscina, etc.)
- `Tags`
  - Estas son descripciones breves que un revisor puede seleccionar para describir el tipo de huésped que era (por ejemplo, solo o en familia), el tipo de habitación que tenía, la duración de la estancia y cómo se envió la reseña.
  - Desafortunadamente, usar estas etiquetas es problemático, consulta la sección a continuación que discute su utilidad.

**Columnas del revisor**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Esto podría ser un factor en un modelo de recomendación, por ejemplo, si pudieras determinar que los revisores más prolíficos con cientos de reseñas eran más propensos a ser negativos que positivos. Sin embargo, el revisor de cualquier reseña en particular no está identificado con un código único, y por lo tanto no puede vincularse a un conjunto de reseñas. Hay 30 revisores con 100 o más reseñas, pero es difícil ver cómo esto puede ayudar al modelo de recomendación.
- `Reviewer_Nationality`
  - Algunas personas podrían pensar que ciertas nacionalidades son más propensas a dar una reseña positiva o negativa debido a una inclinación nacional. Ten cuidado al construir este tipo de puntos de vista anecdóticos en tus modelos. Estos son estereotipos nacionales (y a veces raciales), y cada revisor fue un individuo que escribió una reseña basada en su experiencia. Esta pudo haber sido filtrada a través de muchas perspectivas, como sus estancias previas en hoteles, la distancia recorrida y su temperamento personal. Pensar que su nacionalidad fue la razón de una puntuación de reseña es difícil de justificar.

##### Ejemplos

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Este no es actualmente un hotel sino un sitio de construcción. Fui aterrorizado desde temprano en la mañana y todo el día con ruidos de construcción inaceptables mientras descansaba después de un largo viaje y trabajaba en la habitación. La gente trabajaba todo el día, es decir, con martillos neumáticos en las habitaciones adyacentes. Pedí un cambio de habitación pero no había ninguna habitación silenciosa disponible. Para empeorar las cosas, me cobraron de más. Me fui en la noche ya que tenía un vuelo muy temprano y recibí una factura adecuada. Un día después, el hotel hizo otro cargo sin mi consentimiento por encima del precio reservado. Es un lugar terrible. No te castigues reservando aquí. | Nada. Lugar terrible. Aléjate.   | Viaje de negocios. Pareja. Habitación doble estándar. Estancia de 2 noches.              |

Como puedes ver, este huésped no tuvo una estancia feliz en este hotel. El hotel tiene una buena puntuación promedio de 7.8 y 1945 reseñas, pero este revisor le dio un 2.5 y escribió 115 palabras sobre lo negativa que fue su estancia. Si no escribió nada en absoluto en la columna Positive_Review, podrías deducir que no hubo nada positivo, pero aún así escribió 7 palabras de advertencia. Si solo contáramos palabras en lugar del significado o sentimiento de las palabras, podríamos tener una visión sesgada de la intención del revisor. Curiosamente, su puntuación de 2.5 es confusa, porque si esa estancia en el hotel fue tan mala, ¿por qué darle algún punto? Al investigar el conjunto de datos de cerca, verás que la puntuación más baja posible es 2.5, no 0. La puntuación más alta posible es 10.

##### Tags

Como se mencionó anteriormente, a primera vista, la idea de usar `Tags` para categorizar los datos tiene sentido. Desafortunadamente, estas etiquetas no están estandarizadas, lo que significa que en un hotel dado, las opciones podrían ser *Habitación individual*, *Habitación doble*, y *Habitación twin*, pero en el siguiente hotel, son *Habitación individual deluxe*, *Habitación clásica queen*, y *Habitación ejecutiva king*. Podrían ser lo mismo, pero hay tantas variaciones que la elección se convierte en:

1. Intentar cambiar todos los términos a un estándar único, lo cual es muy difícil, porque no está claro cuál sería el camino de conversión en cada caso (por ejemplo, *Habitación individual clásica* se mapea a *Habitación individual* pero *Habitación superior queen con vista al jardín del patio o a la ciudad* es mucho más difícil de mapear).

1. Podemos tomar un enfoque de NLP y medir la frecuencia de ciertos términos como *Solo*, *Viajero de negocios*, o *Familia con niños pequeños* según se aplican a cada hotel, y factorizar eso en la recomendación.

Las etiquetas suelen ser (pero no siempre) un solo campo que contiene una lista de 5 a 6 valores separados por comas que se alinean con *Tipo de viaje*, *Tipo de huéspedes*, *Tipo de habitación*, *Número de noches*, y *Tipo de dispositivo en el que se envió la reseña*. Sin embargo, debido a que algunos revisores no completan cada campo (pueden dejar uno en blanco), los valores no siempre están en el mismo orden.

Como ejemplo, toma *Tipo de grupo*. Hay 1025 posibilidades únicas en este campo en la columna `Tags`, y desafortunadamente solo algunas de ellas se refieren a un grupo (algunas son el tipo de habitación, etc.). Si filtras solo las que mencionan familia, los resultados contienen muchos resultados del tipo *Habitación familiar*. Si incluyes el término *con*, es decir, cuentas los valores *Familia con*, los resultados son mejores, con más de 80,000 de los 515,000 resultados que contienen la frase "Familia con niños pequeños" o "Familia con niños mayores".

Esto significa que la columna de etiquetas no es completamente inútil para nosotros, pero requerirá algo de trabajo para hacerla útil.

##### Puntuación promedio del hotel

Hay una serie de rarezas o discrepancias con el conjunto de datos que no puedo resolver, pero se ilustran aquí para que estés al tanto de ellas al construir tus modelos. Si lo resuelves, por favor háznoslo saber en la sección de discusión.

El conjunto de datos tiene las siguientes columnas relacionadas con la puntuación promedio y el número de reseñas:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

El único hotel con más reseñas en este conjunto de datos es *Britannia International Hotel Canary Wharf* con 4789 reseñas de 515,000. Pero si miramos el valor de `Total_Number_of_Reviews` para este hotel, es 9086. Podrías deducir que hay muchas más puntuaciones sin reseñas, así que tal vez deberíamos sumar el valor de la columna `Additional_Number_of_Scoring`. Ese valor es 2682, y sumándolo a 4789 obtenemos 7471, que aún está 1615 por debajo de `Total_Number_of_Reviews`.

Si tomas la columna `Average_Score`, podrías deducir que es el promedio de las reseñas en el conjunto de datos, pero la descripción de Kaggle es "*Puntuación promedio del hotel, calculada en base al último comentario del último año*". Eso no parece muy útil, pero podemos calcular nuestro propio promedio basado en las puntuaciones de las reseñas en el conjunto de datos. Usando el mismo hotel como ejemplo, la puntuación promedio del hotel se da como 7.1 pero la puntuación calculada (promedio de las puntuaciones de los revisores *en* el conjunto de datos) es 6.8. Esto es cercano, pero no el mismo valor, y solo podemos suponer que las puntuaciones dadas en las reseñas de `Additional_Number_of_Scoring` aumentaron el promedio a 7.1. Desafortunadamente, sin forma de probar o demostrar esa afirmación, es difícil usar o confiar en `Average_Score`, `Additional_Number_of_Scoring` y `Total_Number_of_Reviews` cuando se basan en, o se refieren a, datos que no tenemos.

Para complicar aún más las cosas, el hotel con el segundo mayor número de reseñas tiene una puntuación promedio calculada de 8.12 y la `Average_Score` del conjunto de datos es 8.1. ¿Es esta puntuación correcta una coincidencia o es el primer hotel una discrepancia?

En la posibilidad de que este hotel pueda ser un caso atípico, y que tal vez la mayoría de los valores coincidan (pero algunos no por alguna razón), escribiremos un programa corto a continuación para explorar los valores en el conjunto de datos y determinar el uso correcto (o no uso) de los valores.
> 🚨 Una nota de precaución  
>  
> Al trabajar con este conjunto de datos, escribirás código que calcula algo a partir del texto sin necesidad de leer o analizar el texto tú mismo. Esta es la esencia del procesamiento de lenguaje natural (NLP), interpretar el significado o el sentimiento sin que un humano tenga que hacerlo. Sin embargo, es posible que leas algunas de las reseñas negativas. Te recomendaría que no lo hagas, porque no es necesario. Algunas de ellas son absurdas o irrelevantes, como reseñas negativas de hoteles que dicen: "El clima no fue bueno", algo que está fuera del control del hotel, o de cualquier persona. Pero también hay un lado oscuro en algunas reseñas. A veces, las reseñas negativas son racistas, sexistas o discriminatorias por edad. Esto es desafortunado pero esperable en un conjunto de datos extraído de un sitio web público. Algunos usuarios dejan reseñas que podrían resultarte desagradables, incómodas o perturbadoras. Es mejor dejar que el código mida el sentimiento en lugar de leerlas tú mismo y sentirte afectado. Dicho esto, es una minoría la que escribe este tipo de cosas, pero existen de todos modos.
## Ejercicio - Exploración de datos
### Cargar los datos

Ya basta de examinar los datos visualmente, ¡ahora escribirás algo de código y obtendrás respuestas! Esta sección utiliza la biblioteca pandas. Tu primera tarea es asegurarte de que puedes cargar y leer los datos en formato CSV. La biblioteca pandas tiene un cargador rápido de CSV, y el resultado se coloca en un dataframe, como en lecciones anteriores. El CSV que estamos cargando tiene más de medio millón de filas, pero solo 17 columnas. Pandas te ofrece muchas formas poderosas de interactuar con un dataframe, incluyendo la capacidad de realizar operaciones en cada fila.

A partir de aquí, en esta lección, habrá fragmentos de código y algunas explicaciones del código, además de una discusión sobre lo que significan los resultados. Usa el archivo _notebook.ipynb_ incluido para tu código.

Comencemos cargando el archivo de datos que usarás:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Ahora que los datos están cargados, podemos realizar algunas operaciones sobre ellos. Mantén este código en la parte superior de tu programa para la siguiente parte.

## Explorar los datos

En este caso, los datos ya están *limpios*, lo que significa que están listos para trabajar y no tienen caracteres en otros idiomas que puedan causar problemas a los algoritmos que esperan solo caracteres en inglés.

✅ Es posible que tengas que trabajar con datos que requieran un procesamiento inicial para formatearlos antes de aplicar técnicas de NLP, pero no en esta ocasión. Si tuvieras que hacerlo, ¿cómo manejarías los caracteres que no están en inglés?

Tómate un momento para asegurarte de que, una vez cargados los datos, puedes explorarlos con código. Es muy fácil querer centrarse en las columnas `Negative_Review` y `Positive_Review`. Estas están llenas de texto natural para que tus algoritmos de NLP lo procesen. ¡Pero espera! Antes de sumergirte en el NLP y el análisis de sentimientos, deberías seguir el código a continuación para verificar si los valores dados en el conjunto de datos coinciden con los valores que calculas con pandas.

## Operaciones con el dataframe

La primera tarea en esta lección es verificar si las siguientes afirmaciones son correctas escribiendo algo de código que examine el dataframe (sin modificarlo).

> Como en muchas tareas de programación, hay varias formas de completarlas, pero un buen consejo es hacerlo de la manera más simple y fácil posible, especialmente si será más fácil de entender cuando vuelvas a este código en el futuro. Con los dataframes, hay una API completa que a menudo tendrá una forma eficiente de hacer lo que necesitas.

Trata las siguientes preguntas como tareas de codificación e intenta responderlas sin mirar la solución.

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas).
2. Calcula el conteo de frecuencia para las nacionalidades de los revisores:
   1. ¿Cuántos valores distintos hay en la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Qué nacionalidad de revisor es la más común en el conjunto de datos (imprime el país y el número de reseñas)?
   3. ¿Cuáles son las siguientes 10 nacionalidades más frecuentes y su conteo de frecuencia?
3. ¿Cuál fue el hotel más reseñado para cada una de las 10 nacionalidades de revisores más frecuentes?
4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia de hotel) en el conjunto de datos?
5. Aunque hay una columna `Average_Score` para cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los revisores en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga ese promedio calculado.
6. ¿Hay hoteles que tengan el mismo `Average_Score` (redondeado a 1 decimal) y `Calc_Average_Score`?
   1. Intenta escribir una función en Python que tome una Serie (fila) como argumento y compare los valores, imprimiendo un mensaje cuando los valores no sean iguales. Luego usa el método `.apply()` para procesar cada fila con la función.
7. Calcula e imprime cuántas filas tienen valores de la columna `Negative_Review` iguales a "No Negative".
8. Calcula e imprime cuántas filas tienen valores de la columna `Positive_Review` iguales a "No Positive".
9. Calcula e imprime cuántas filas tienen valores de la columna `Positive_Review` iguales a "No Positive" **y** valores de la columna `Negative_Review` iguales a "No Negative".

### Respuestas en código

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcula el conteo de frecuencia para las nacionalidades de los revisores:

   1. ¿Cuántos valores distintos hay en la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Qué nacionalidad de revisor es la más común en el conjunto de datos (imprime el país y el número de reseñas)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. ¿Cuáles son las siguientes 10 nacionalidades más frecuentes y su conteo de frecuencia?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. ¿Cuál fue el hotel más reseñado para cada una de las 10 nacionalidades de revisores más frecuentes?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia de hotel) en el conjunto de datos?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |

   Puedes notar que los resultados *contados en el conjunto de datos* no coinciden con el valor en `Total_Number_of_Reviews`. No está claro si este valor en el conjunto de datos representa el número total de reseñas que tuvo el hotel, pero no todas fueron extraídas, o algún otro cálculo. `Total_Number_of_Reviews` no se utiliza en el modelo debido a esta falta de claridad.

5. Aunque hay una columna `Average_Score` para cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los revisores en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga ese promedio calculado. Imprime las columnas `Hotel_Name`, `Average_Score` y `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   También puedes preguntarte sobre el valor de `Average_Score` y por qué a veces es diferente del puntaje promedio calculado. Como no podemos saber por qué algunos valores coinciden, pero otros tienen una diferencia, lo más seguro en este caso es usar los puntajes de las reseñas que tenemos para calcular el promedio nosotros mismos. Dicho esto, las diferencias suelen ser muy pequeñas, aquí están los hoteles con la mayor desviación entre el promedio del conjunto de datos y el promedio calculado:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Con solo 1 hotel teniendo una diferencia de puntaje mayor a 1, significa que probablemente podemos ignorar la diferencia y usar el puntaje promedio calculado.

6. Calcula e imprime cuántas filas tienen valores de la columna `Negative_Review` iguales a "No Negative".

7. Calcula e imprime cuántas filas tienen valores de la columna `Positive_Review` iguales a "No Positive".

8. Calcula e imprime cuántas filas tienen valores de la columna `Positive_Review` iguales a "No Positive" **y** valores de la columna `Negative_Review` iguales a "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Otra forma

Otra forma de contar elementos sin Lambdas, y usar sum para contar las filas:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Puede que hayas notado que hay 127 filas que tienen valores "No Negative" y "No Positive" en las columnas `Negative_Review` y `Positive_Review`, respectivamente. Esto significa que el revisor dio al hotel un puntaje numérico, pero se negó a escribir una reseña positiva o negativa. Afortunadamente, esta es una pequeña cantidad de filas (127 de 515738, o 0.02%), por lo que probablemente no sesgará nuestro modelo o resultados en ninguna dirección en particular, pero podrías no haber esperado que un conjunto de datos de reseñas tuviera filas sin reseñas, por lo que vale la pena explorar los datos para descubrir filas como esta.

Ahora que has explorado el conjunto de datos, en la próxima lección filtrarás los datos y agregarás algo de análisis de sentimientos.

---
## 🚀Desafío

Esta lección demuestra, como vimos en lecciones anteriores, lo críticamente importante que es entender tus datos y sus peculiaridades antes de realizar operaciones sobre ellos. Los datos basados en texto, en particular, requieren un escrutinio cuidadoso. Explora varios conjuntos de datos ricos en texto y ve si puedes descubrir áreas que podrían introducir sesgos o sentimientos distorsionados en un modelo.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Toma [este Learning Path sobre NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descubrir herramientas que puedes probar al construir modelos basados en texto y voz.

## Tarea 

[NLTK](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-04T22:29:39+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "es"
}
-->
# Análisis de sentimientos con reseñas de hoteles

Ahora que has explorado el conjunto de datos en detalle, es momento de filtrar las columnas y luego usar técnicas de procesamiento de lenguaje natural (NLP) en el conjunto de datos para obtener nuevas perspectivas sobre los hoteles.

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

### Operaciones de filtrado y análisis de sentimientos

Como probablemente hayas notado, el conjunto de datos tiene algunos problemas. Algunas columnas están llenas de información inútil, otras parecen incorrectas. Si son correctas, no está claro cómo se calcularon, y las respuestas no pueden ser verificadas de manera independiente con tus propios cálculos.

## Ejercicio: un poco más de procesamiento de datos

Limpia los datos un poco más. Agrega columnas que serán útiles más adelante, cambia los valores en otras columnas y elimina ciertas columnas por completo.

1. Procesamiento inicial de columnas

   1. Elimina `lat` y `lng`.

   2. Reemplaza los valores de `Hotel_Address` con los siguientes valores (si la dirección contiene el nombre de la ciudad y el país, cámbialo por solo la ciudad y el país).

      Estas son las únicas ciudades y países en el conjunto de datos:

      Ámsterdam, Países Bajos

      Barcelona, España

      Londres, Reino Unido

      Milán, Italia

      París, Francia

      Viena, Austria 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Ahora puedes consultar datos a nivel de país:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Ámsterdam, Países Bajos |    105     |
      | Barcelona, España       |    211     |
      | Londres, Reino Unido    |    400     |
      | Milán, Italia           |    162     |
      | París, Francia          |    458     |
      | Viena, Austria          |    158     |

2. Procesar columnas de meta-reseñas de hoteles

   1. Elimina `Additional_Number_of_Scoring`.

   2. Reemplaza `Total_Number_of_Reviews` con el número total de reseñas para ese hotel que realmente están en el conjunto de datos.

   3. Reemplaza `Average_Score` con nuestro propio puntaje calculado.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Procesar columnas de reseñas

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` y `days_since_review`.

   2. Conserva `Reviewer_Score`, `Negative_Review` y `Positive_Review` tal como están.

   3. Conserva `Tags` por ahora.

      - Realizaremos algunas operaciones de filtrado adicionales en las etiquetas en la siguiente sección y luego las eliminaremos.

4. Procesar columnas de los revisores

   1. Elimina `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Conserva `Reviewer_Nationality`.

### Columnas de etiquetas

La columna `Tag` es problemática ya que es una lista (en forma de texto) almacenada en la columna. Desafortunadamente, el orden y el número de subsecciones en esta columna no siempre son los mismos. Es difícil para un humano identificar las frases correctas de interés, porque hay 515,000 filas y 1427 hoteles, y cada uno tiene opciones ligeramente diferentes que un revisor podría elegir. Aquí es donde el NLP brilla. Puedes escanear el texto y encontrar las frases más comunes, y contarlas.

Desafortunadamente, no estamos interesados en palabras individuales, sino en frases de varias palabras (por ejemplo, *Viaje de negocios*). Ejecutar un algoritmo de distribución de frecuencia de frases en tantos datos (6762646 palabras) podría tomar un tiempo extraordinario, pero sin mirar los datos, parecería que es un gasto necesario. Aquí es donde el análisis exploratorio de datos resulta útil, porque al haber visto una muestra de las etiquetas como `[' Viaje de negocios  ', ' Viajero solo ', ' Habitación individual ', ' Estancia de 5 noches ', ' Enviado desde un dispositivo móvil ']`, puedes comenzar a preguntarte si es posible reducir significativamente el procesamiento que tienes que hacer. Afortunadamente, lo es, pero primero necesitas seguir algunos pasos para determinar las etiquetas de interés.

### Filtrar etiquetas

Recuerda que el objetivo del conjunto de datos es agregar sentimientos y columnas que te ayuden a elegir el mejor hotel (para ti o tal vez para un cliente que te encargue crear un bot de recomendación de hoteles). Debes preguntarte si las etiquetas son útiles o no en el conjunto de datos final. Aquí hay una interpretación (si necesitaras el conjunto de datos por otras razones, diferentes etiquetas podrían permanecer o salir de la selección):

1. El tipo de viaje es relevante y debería permanecer.
2. El tipo de grupo de huéspedes es importante y debería permanecer.
3. El tipo de habitación, suite o estudio en el que se hospedó el huésped es irrelevante (todos los hoteles tienen básicamente las mismas habitaciones).
4. El dispositivo desde el cual se envió la reseña es irrelevante.
5. El número de noches que el revisor se hospedó *podría* ser relevante si atribuyes estancias más largas con que les gustó más el hotel, pero es poco probable y probablemente irrelevante.

En resumen, **conserva 2 tipos de etiquetas y elimina las demás**.

Primero, no quieres contar las etiquetas hasta que estén en un formato mejor, lo que significa eliminar los corchetes y las comillas. Puedes hacer esto de varias maneras, pero quieres la más rápida ya que podría tomar mucho tiempo procesar muchos datos. Afortunadamente, pandas tiene una forma fácil de realizar cada uno de estos pasos.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada etiqueta se convierte en algo como: `Viaje de negocios, Viajero solo, Habitación individual, Estancia de 5 noches, Enviado desde un dispositivo móvil`.

A continuación, encontramos un problema. Algunas reseñas, o filas, tienen 5 columnas, otras 3, otras 6. Esto es resultado de cómo se creó el conjunto de datos y es difícil de corregir. Quieres obtener un conteo de frecuencia de cada frase, pero están en diferentes órdenes en cada reseña, por lo que el conteo podría estar incorrecto y un hotel podría no recibir una etiqueta que merecía.

En cambio, usarás el orden diferente a tu favor, porque cada etiqueta es de varias palabras pero también está separada por una coma. La forma más sencilla de hacer esto es crear 6 columnas temporales con cada etiqueta insertada en la columna correspondiente a su orden en la etiqueta. Luego puedes fusionar las 6 columnas en una gran columna y ejecutar el método `value_counts()` en la columna resultante. Al imprimir eso, verás que había 2428 etiquetas únicas. Aquí hay una pequeña muestra:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Viaje de ocio                  | 417778 |
| Enviado desde un dispositivo móvil | 307640 |
| Pareja                         | 252294 |
| Estancia de 1 noche            | 193645 |
| Estancia de 2 noches           | 133937 |
| Viajero solo                   | 108545 |
| Estancia de 3 noches           | 95821  |
| Viaje de negocios              | 82939  |
| Grupo                          | 65392  |
| Familia con niños pequeños     | 61015  |
| Estancia de 4 noches           | 47817  |
| Habitación doble               | 35207  |
| Habitación doble estándar      | 32248  |
| Habitación doble superior      | 31393  |
| Familia con niños mayores      | 26349  |
| Habitación doble deluxe        | 24823  |
| Habitación doble o twin        | 22393  |
| Estancia de 5 noches           | 20845  |
| Habitación doble estándar o twin | 17483  |
| Habitación doble clásica       | 16989  |
| Habitación doble superior o twin | 13570 |
| 2 habitaciones                 | 12393  |

Algunas de las etiquetas comunes como `Enviado desde un dispositivo móvil` no nos son útiles, por lo que podría ser inteligente eliminarlas antes de contar la ocurrencia de frases, pero es una operación tan rápida que puedes dejarlas y simplemente ignorarlas.

### Eliminar las etiquetas de duración de la estancia

Eliminar estas etiquetas es el paso 1, reduce ligeramente el número total de etiquetas a considerar. Nota que no las eliminas del conjunto de datos, solo eliges eliminarlas de la consideración como valores para contar/conservar en el conjunto de reseñas.

| Duración de la estancia | Count  |
| ----------------------- | ------ |
| Estancia de 1 noche     | 193645 |
| Estancia de 2 noches    | 133937 |
| Estancia de 3 noches    | 95821  |
| Estancia de 4 noches    | 47817  |
| Estancia de 5 noches    | 20845  |
| Estancia de 6 noches    | 9776   |
| Estancia de 7 noches    | 7399   |
| Estancia de 8 noches    | 2502   |
| Estancia de 9 noches    | 1293   |
| ...                     | ...    |

Hay una gran variedad de habitaciones, suites, estudios, apartamentos y demás. Todos significan aproximadamente lo mismo y no son relevantes para ti, así que elimínalos de la consideración.

| Tipo de habitación              | Count |
| ------------------------------- | ----- |
| Habitación doble                | 35207 |
| Habitación doble estándar       | 32248 |
| Habitación doble superior       | 31393 |
| Habitación doble deluxe         | 24823 |
| Habitación doble o twin         | 22393 |
| Habitación doble estándar o twin | 17483 |
| Habitación doble clásica        | 16989 |
| Habitación doble superior o twin | 13570 |

Finalmente, y esto es encantador (porque no tomó mucho procesamiento en absoluto), te quedarás con las siguientes etiquetas *útiles*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Viaje de ocio                                 | 417778 |
| Pareja                                        | 252294 |
| Viajero solo                                  | 108545 |
| Viaje de negocios                             | 82939  |
| Grupo (combinado con Viajeros con amigos)     | 67535  |
| Familia con niños pequeños                    | 61015  |
| Familia con niños mayores                     | 26349  |
| Con una mascota                               | 1405   |

Podrías argumentar que `Viajeros con amigos` es lo mismo que `Grupo` más o menos, y sería justo combinarlos como se muestra arriba. El código para identificar las etiquetas correctas está en [el cuaderno de etiquetas](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

El paso final es crear nuevas columnas para cada una de estas etiquetas. Luego, para cada fila de reseña, si la columna `Tag` coincide con una de las nuevas columnas, agrega un 1, si no, agrega un 0. El resultado final será un conteo de cuántos revisores eligieron este hotel (en conjunto) para, por ejemplo, negocios vs ocio, o para llevar una mascota, y esta es información útil al recomendar un hotel.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Guarda tu archivo

Finalmente, guarda el conjunto de datos tal como está ahora con un nuevo nombre.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operaciones de análisis de sentimientos

En esta sección final, aplicarás análisis de sentimientos a las columnas de reseñas y guardarás los resultados en un conjunto de datos.

## Ejercicio: cargar y guardar los datos filtrados

Nota que ahora estás cargando el conjunto de datos filtrado que se guardó en la sección anterior, **no** el conjunto de datos original.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Eliminar palabras vacías

Si ejecutaras análisis de sentimientos en las columnas de reseñas negativas y positivas, podría tomar mucho tiempo. Probado en una laptop de prueba potente con CPU rápida, tomó entre 12 y 14 minutos dependiendo de la biblioteca de análisis de sentimientos utilizada. Ese es un tiempo (relativamente) largo, por lo que vale la pena investigar si se puede acelerar.

Eliminar palabras vacías, o palabras comunes en inglés que no cambian el sentimiento de una oración, es el primer paso. Al eliminarlas, el análisis de sentimientos debería ejecutarse más rápido, pero no ser menos preciso (ya que las palabras vacías no afectan el sentimiento, pero sí ralentizan el análisis).

La reseña negativa más larga tenía 395 palabras, pero después de eliminar las palabras vacías, tiene 195 palabras.

Eliminar las palabras vacías también es una operación rápida, eliminarlas de 2 columnas de reseñas en más de 515,000 filas tomó 3.3 segundos en el dispositivo de prueba. Podría tomar un poco más o menos tiempo para ti dependiendo de la velocidad de tu CPU, RAM, si tienes un SSD o no, y algunos otros factores. La relativa brevedad de la operación significa que si mejora el tiempo de análisis de sentimientos, entonces vale la pena hacerlo.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Realizar análisis de sentimientos

Ahora deberías calcular el análisis de sentimientos para las columnas de reseñas negativas y positivas, y almacenar el resultado en 2 nuevas columnas. La prueba del sentimiento será compararlo con la puntuación del revisor para la misma reseña. Por ejemplo, si el análisis de sentimientos piensa que la reseña negativa tenía un sentimiento de 1 (sentimiento extremadamente positivo) y un sentimiento de reseña positiva de 1, pero el revisor dio al hotel la puntuación más baja posible, entonces o el texto de la reseña no coincide con la puntuación, o el analizador de sentimientos no pudo reconocer correctamente el sentimiento. Deberías esperar que algunas puntuaciones de sentimiento sean completamente incorrectas, y a menudo eso será explicable, por ejemplo, la reseña podría ser extremadamente sarcástica: "Por supuesto que AMÉ dormir en una habitación sin calefacción" y el analizador de sentimientos piensa que eso es un sentimiento positivo, aunque un humano que lo lea sabría que es sarcasmo.
NLTK ofrece diferentes analizadores de sentimientos para aprender, y puedes sustituirlos y ver si el análisis de sentimientos es más o menos preciso. Aquí se utiliza el análisis de sentimientos VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Un modelo parsimonioso basado en reglas para el análisis de sentimientos de textos en redes sociales. Octava Conferencia Internacional sobre Blogs y Redes Sociales (ICWSM-14). Ann Arbor, MI, junio de 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Más adelante en tu programa, cuando estés listo para calcular el sentimiento, puedes aplicarlo a cada reseña de la siguiente manera:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Esto toma aproximadamente 120 segundos en mi computadora, pero puede variar en cada equipo. Si deseas imprimir los resultados y verificar si el sentimiento coincide con la reseña:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Lo último que debes hacer con el archivo antes de usarlo en el desafío es ¡guardarlo! También deberías considerar reorganizar todas tus nuevas columnas para que sean fáciles de trabajar (para una persona, es un cambio estético).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Debes ejecutar todo el código del [notebook de análisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (después de haber ejecutado [el notebook de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) para generar el archivo Hotel_Reviews_Filtered.csv).

Para repasar, los pasos son:

1. El archivo del conjunto de datos original **Hotel_Reviews.csv** se explora en la lección anterior con [el notebook explorador](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv se filtra con [el notebook de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), resultando en **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv se procesa con [el notebook de análisis de sentimientos](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), resultando en **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv en el desafío de NLP a continuación

### Conclusión

Cuando comenzaste, tenías un conjunto de datos con columnas y datos, pero no todo podía ser verificado o utilizado. Has explorado los datos, filtrado lo que no necesitas, convertido etiquetas en algo útil, calculado tus propios promedios, añadido algunas columnas de sentimientos y, con suerte, aprendido cosas interesantes sobre el procesamiento de texto natural.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Desafío

Ahora que tienes tu conjunto de datos analizado para sentimientos, intenta usar las estrategias que has aprendido en este curso (¿quizás agrupamiento?) para determinar patrones relacionados con los sentimientos.

## Revisión y autoestudio

Toma [este módulo de aprendizaje](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender más y usar diferentes herramientas para explorar sentimientos en texto.

## Tarea

[Prueba con un conjunto de datos diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---
