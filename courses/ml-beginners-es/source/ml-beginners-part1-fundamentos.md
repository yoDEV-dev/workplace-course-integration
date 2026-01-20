# ML para Principiantes 🤖
## Parte 1: Fundamentos
**Introducción al Machine Learning y Regresión**

---


# Introducción al Machine Learning

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "cf8ecc83f28e5b98051d2179eca08e08",
  "translation_date": "2025-09-03T23:26:11+00:00",
  "source_file": "1-Introduction/README.md",
  "language_code": "es"
}
-->
# Introducción al aprendizaje automático

En esta sección del plan de estudios, se te presentarán los conceptos básicos que sustentan el campo del aprendizaje automático, qué es, y aprenderás sobre su historia y las técnicas que los investigadores utilizan para trabajar con él. ¡Exploremos juntos este nuevo mundo del aprendizaje automático!

![globo](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/globe.59f26379ceb40428.webp)
> Foto de <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Bill Oxford</a> en <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
### Lecciones

1. [Introducción al aprendizaje automático](1-intro-to-ML/README.md)
1. [La historia del aprendizaje automático y la inteligencia artificial](2-history-of-ML/README.md)
1. [Equidad y aprendizaje automático](3-fairness/README.md)
1. [Técnicas de aprendizaje automático](4-techniques-of-ML/README.md)

### Créditos

"Introducción al Aprendizaje Automático" fue escrito con ♥️ por un equipo de personas que incluye [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan), [Ornella Altunyan](https://twitter.com/ornelladotcom) y [Jen Looper](https://twitter.com/jenlooper)

"La Historia del Aprendizaje Automático" fue escrita con ♥️ por [Jen Looper](https://twitter.com/jenlooper) y [Amy Boyd](https://twitter.com/AmyKateNicho)

"Equidad y Aprendizaje Automático" fue escrito con ♥️ por [Tomomi Imura](https://twitter.com/girliemac) 

"Técnicas de Aprendizaje Automático" fue escrito con ♥️ por [Jen Looper](https://twitter.com/jenlooper) y [Chris Noring](https://twitter.com/softchris)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-04T22:21:40+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "es"
}
-->
# Introducción al aprendizaje automático

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para principiantes - Introducción al aprendizaje automático para principiantes](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML para principiantes - Introducción al aprendizaje automático para principiantes")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre esta lección.

¡Bienvenido a este curso sobre aprendizaje automático clásico para principiantes! Ya sea que seas completamente nuevo en este tema o un practicante experimentado de ML que busca repasar un área, ¡nos alegra que te unas a nosotros! Queremos crear un punto de partida amigable para tu estudio de ML y estaremos encantados de evaluar, responder e incorporar tus [comentarios](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introducción al aprendizaje automático](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introducción al aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT introduce el aprendizaje automático.

---
## Comenzando con el aprendizaje automático

Antes de comenzar con este plan de estudios, necesitas tener tu computadora configurada y lista para ejecutar notebooks de manera local.

- **Configura tu máquina con estos videos**. Usa los siguientes enlaces para aprender [cómo instalar Python](https://youtu.be/CXZYvNRIAKM) en tu sistema y [configurar un editor de texto](https://youtu.be/EU8eayHWoZg) para el desarrollo.
- **Aprende Python**. También se recomienda tener un entendimiento básico de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un lenguaje de programación útil para científicos de datos que utilizamos en este curso.
- **Aprende Node.js y JavaScript**. También utilizamos JavaScript algunas veces en este curso al construir aplicaciones web, por lo que necesitarás tener [node](https://nodejs.org) y [npm](https://www.npmjs.com/) instalados, así como [Visual Studio Code](https://code.visualstudio.com/) disponible para el desarrollo tanto en Python como en JavaScript.
- **Crea una cuenta de GitHub**. Ya que nos encontraste aquí en [GitHub](https://github.com), es posible que ya tengas una cuenta, pero si no, crea una y luego haz un fork de este plan de estudios para usarlo por tu cuenta. (También puedes darnos una estrella 😊).
- **Explora Scikit-learn**. Familiarízate con [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un conjunto de bibliotecas de ML que referenciamos en estas lecciones.

---
## ¿Qué es el aprendizaje automático?

El término 'aprendizaje automático' es uno de los más populares y frecuentemente utilizados hoy en día. Existe una posibilidad no trivial de que hayas escuchado este término al menos una vez si tienes algún tipo de familiaridad con la tecnología, sin importar el área en la que trabajes. Sin embargo, la mecánica del aprendizaje automático es un misterio para la mayoría de las personas. Para un principiante en aprendizaje automático, el tema puede parecer abrumador a veces. Por lo tanto, es importante entender qué es realmente el aprendizaje automático y aprender sobre él paso a paso, a través de ejemplos prácticos.

---
## La curva de expectativas

![curva de expectativas de ML](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends muestra la reciente 'curva de expectativas' del término 'aprendizaje automático'.

---
## Un universo misterioso

Vivimos en un universo lleno de misterios fascinantes. Grandes científicos como Stephen Hawking, Albert Einstein y muchos más han dedicado sus vidas a buscar información significativa que revele los misterios del mundo que nos rodea. Esta es la condición humana de aprender: un niño humano aprende cosas nuevas y descubre la estructura de su mundo año tras año mientras crece hasta la adultez.

---
## El cerebro de un niño

El cerebro y los sentidos de un niño perciben los hechos de su entorno y gradualmente aprenden los patrones ocultos de la vida que ayudan al niño a crear reglas lógicas para identificar patrones aprendidos. El proceso de aprendizaje del cerebro humano hace que los humanos sean la criatura más sofisticada de este mundo. Aprender continuamente descubriendo patrones ocultos y luego innovando sobre esos patrones nos permite mejorar continuamente a lo largo de nuestra vida. Esta capacidad de aprendizaje y evolución está relacionada con un concepto llamado [plasticidad cerebral](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, podemos establecer algunas similitudes motivacionales entre el proceso de aprendizaje del cerebro humano y los conceptos del aprendizaje automático.

---
## El cerebro humano

El [cerebro humano](https://www.livescience.com/29365-human-brain.html) percibe cosas del mundo real, procesa la información percibida, toma decisiones racionales y realiza ciertas acciones según las circunstancias. Esto es lo que llamamos comportarse de manera inteligente. Cuando programamos una réplica del proceso de comportamiento inteligente en una máquina, se llama inteligencia artificial (IA).

---
## Algunos términos

Aunque los términos pueden confundirse, el aprendizaje automático (ML) es un subconjunto importante de la inteligencia artificial. **ML se ocupa de usar algoritmos especializados para descubrir información significativa y encontrar patrones ocultos a partir de datos percibidos para corroborar el proceso de toma de decisiones racionales**.

---
## IA, ML, Aprendizaje profundo

![IA, ML, aprendizaje profundo, ciencia de datos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Un diagrama que muestra las relaciones entre IA, ML, aprendizaje profundo y ciencia de datos. Infografía por [Jen Looper](https://twitter.com/jenlooper) inspirada en [este gráfico](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining).

---
## Conceptos a cubrir

En este plan de estudios, vamos a cubrir solo los conceptos básicos del aprendizaje automático que un principiante debe conocer. Cubrimos lo que llamamos 'aprendizaje automático clásico', principalmente utilizando Scikit-learn, una excelente biblioteca que muchos estudiantes usan para aprender los fundamentos. Para entender conceptos más amplios de inteligencia artificial o aprendizaje profundo, es indispensable tener un conocimiento fundamental sólido del aprendizaje automático, y por eso queremos ofrecerlo aquí.

---
## En este curso aprenderás:

- conceptos básicos del aprendizaje automático
- la historia del ML
- ML y equidad
- técnicas de regresión en ML
- técnicas de clasificación en ML
- técnicas de agrupamiento en ML
- técnicas de procesamiento de lenguaje natural en ML
- técnicas de predicción de series temporales en ML
- aprendizaje por refuerzo
- aplicaciones reales del ML

---
## Lo que no cubriremos

- aprendizaje profundo
- redes neuronales
- IA

Para ofrecer una mejor experiencia de aprendizaje, evitaremos las complejidades de las redes neuronales, el 'aprendizaje profundo' - construcción de modelos con muchas capas utilizando redes neuronales - y la IA, que discutiremos en un plan de estudios diferente. También ofreceremos un próximo plan de estudios sobre ciencia de datos para centrarnos en ese aspecto de este campo más amplio.

---
## ¿Por qué estudiar aprendizaje automático?

El aprendizaje automático, desde una perspectiva de sistemas, se define como la creación de sistemas automatizados que pueden aprender patrones ocultos a partir de datos para ayudar en la toma de decisiones inteligentes.

Esta motivación está vagamente inspirada en cómo el cerebro humano aprende ciertas cosas basándose en los datos que percibe del mundo exterior.

✅ Piensa por un momento por qué una empresa querría intentar usar estrategias de aprendizaje automático en lugar de crear un motor basado en reglas codificadas.

---
## Aplicaciones del aprendizaje automático

Las aplicaciones del aprendizaje automático están ahora casi en todas partes y son tan ubicuas como los datos que fluyen en nuestras sociedades, generados por nuestros teléfonos inteligentes, dispositivos conectados y otros sistemas. Considerando el inmenso potencial de los algoritmos de aprendizaje automático de última generación, los investigadores han estado explorando su capacidad para resolver problemas reales multidimensionales y multidisciplinarios con grandes resultados positivos.

---
## Ejemplos de ML aplicado

**Puedes usar el aprendizaje automático de muchas maneras**:

- Para predecir la probabilidad de una enfermedad a partir del historial médico o informes de un paciente.
- Para aprovechar los datos meteorológicos y predecir eventos climáticos.
- Para entender el sentimiento de un texto.
- Para detectar noticias falsas y detener la propagación de propaganda.

Finanzas, economía, ciencias de la tierra, exploración espacial, ingeniería biomédica, ciencias cognitivas e incluso áreas de las humanidades han adaptado el aprendizaje automático para resolver los arduos problemas de procesamiento de datos en sus dominios.

---
## Conclusión

El aprendizaje automático automatiza el proceso de descubrimiento de patrones al encontrar información significativa a partir de datos reales o generados. Ha demostrado ser altamente valioso en aplicaciones empresariales, de salud y financieras, entre otras.

En un futuro cercano, entender los fundamentos del aprendizaje automático será imprescindible para personas de cualquier área debido a su adopción generalizada.

---
# 🚀 Desafío

Dibuja, en papel o usando una aplicación en línea como [Excalidraw](https://excalidraw.com/), tu comprensión de las diferencias entre IA, ML, aprendizaje profundo y ciencia de datos. Agrega algunas ideas sobre los problemas que cada una de estas técnicas es buena para resolver.

# [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisión y autoestudio

Para aprender más sobre cómo trabajar con algoritmos de ML en la nube, sigue este [Camino de Aprendizaje](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Toma un [Camino de Aprendizaje](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sobre los fundamentos del ML.

---
# Tarea

[Ponte en marcha](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-04T22:22:07+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "es"
}
-->
# Historia del aprendizaje automático

![Resumen de la historia del aprendizaje automático en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-history.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para principiantes - Historia del aprendizaje automático](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML para principiantes - Historia del aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre esta lección.

En esta lección, repasaremos los hitos más importantes en la historia del aprendizaje automático y la inteligencia artificial.

La historia de la inteligencia artificial (IA) como campo está entrelazada con la historia del aprendizaje automático, ya que los algoritmos y avances computacionales que sustentan el aprendizaje automático contribuyeron al desarrollo de la IA. Es útil recordar que, aunque estos campos como áreas de investigación distintas comenzaron a cristalizar en la década de 1950, importantes [descubrimientos algorítmicos, estadísticos, matemáticos, computacionales y técnicos](https://wikipedia.org/wiki/Timeline_of_machine_learning) precedieron y se superpusieron a esta era. De hecho, las personas han estado reflexionando sobre estas cuestiones durante [cientos de años](https://wikipedia.org/wiki/History_of_artificial_intelligence): este artículo analiza los fundamentos intelectuales históricos de la idea de una 'máquina pensante'.

---
## Descubrimientos notables

- 1763, 1812 [Teorema de Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) y sus predecesores. Este teorema y sus aplicaciones son fundamentales para la inferencia, describiendo la probabilidad de que ocurra un evento basado en conocimientos previos.
- 1805 [Teoría de los mínimos cuadrados](https://wikipedia.org/wiki/Least_squares) por el matemático francés Adrien-Marie Legendre. Esta teoría, que aprenderás en nuestra unidad de Regresión, ayuda en el ajuste de datos.
- 1913 [Cadenas de Markov](https://wikipedia.org/wiki/Markov_chain), nombradas en honor al matemático ruso Andrey Markov, se utilizan para describir una secuencia de eventos posibles basada en un estado previo.
- 1957 [Perceptrón](https://wikipedia.org/wiki/Perceptron), un tipo de clasificador lineal inventado por el psicólogo estadounidense Frank Rosenblatt que sustenta los avances en el aprendizaje profundo.

---

- 1967 [Vecino más cercano](https://wikipedia.org/wiki/Nearest_neighbor) es un algoritmo originalmente diseñado para trazar rutas. En un contexto de aprendizaje automático, se utiliza para detectar patrones.
- 1970 [Retropropagación](https://wikipedia.org/wiki/Backpropagation) se utiliza para entrenar [redes neuronales feedforward](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Redes neuronales recurrentes](https://wikipedia.org/wiki/Recurrent_neural_network) son redes neuronales artificiales derivadas de las redes neuronales feedforward que crean gráficos temporales.

✅ Investiga un poco. ¿Qué otras fechas destacan como fundamentales en la historia del aprendizaje automático y la IA?

---
## 1950: Máquinas que piensan

Alan Turing, una persona verdaderamente extraordinaria que fue votada [por el público en 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) como el mejor científico del siglo XX, es reconocido por ayudar a sentar las bases del concepto de una 'máquina que puede pensar'. Se enfrentó a detractores y a su propia necesidad de evidencia empírica de este concepto en parte creando el [Test de Turing](https://www.bbc.com/news/technology-18475646), que explorarás en nuestras lecciones de procesamiento de lenguaje natural.

---
## 1956: Proyecto de investigación de verano en Dartmouth

"El Proyecto de investigación de verano en Dartmouth sobre inteligencia artificial fue un evento fundamental para la inteligencia artificial como campo", y fue aquí donde se acuñó el término 'inteligencia artificial' ([fuente](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Cada aspecto del aprendizaje o cualquier otra característica de la inteligencia puede, en principio, describirse tan precisamente que se pueda construir una máquina que lo simule.

---

El investigador principal, el profesor de matemáticas John McCarthy, esperaba "proceder sobre la base de la conjetura de que cada aspecto del aprendizaje o cualquier otra característica de la inteligencia puede, en principio, describirse tan precisamente que se pueda construir una máquina que lo simule". Los participantes incluyeron a otro destacado en el campo, Marvin Minsky.

El taller es reconocido por haber iniciado y fomentado varias discusiones, incluyendo "el auge de los métodos simbólicos, sistemas enfocados en dominios limitados (primeros sistemas expertos) y sistemas deductivos frente a sistemas inductivos". ([fuente](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Los años dorados"

Desde la década de 1950 hasta mediados de los años 70, el optimismo era alto con la esperanza de que la IA pudiera resolver muchos problemas. En 1967, Marvin Minsky afirmó con confianza que "Dentro de una generación... el problema de crear 'inteligencia artificial' estará sustancialmente resuelto". (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La investigación en procesamiento de lenguaje natural floreció, la búsqueda se refinó y se hizo más poderosa, y se creó el concepto de 'micro-mundos', donde se completaban tareas simples utilizando instrucciones en lenguaje sencillo.

---

La investigación fue bien financiada por agencias gubernamentales, se lograron avances en computación y algoritmos, y se construyeron prototipos de máquinas inteligentes. Algunas de estas máquinas incluyen:

* [Shakey el robot](https://wikipedia.org/wiki/Shakey_the_robot), que podía maniobrar y decidir cómo realizar tareas 'inteligentemente'.

    ![Shakey, un robot inteligente](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey en 1972

---

* Eliza, un primer 'chatterbot', podía conversar con personas y actuar como un 'terapeuta' primitivo. Aprenderás más sobre Eliza en las lecciones de procesamiento de lenguaje natural.

    ![Eliza, un bot](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/2-history-of-ML/images/eliza.png)
    > Una versión de Eliza, un chatbot

---

* "Blocks world" fue un ejemplo de un micro-mundo donde los bloques podían apilarse y ordenarse, y se podían probar experimentos para enseñar a las máquinas a tomar decisiones. Los avances construidos con bibliotecas como [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ayudaron a impulsar el procesamiento de lenguaje.

    [![blocks world con SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world con SHRDLU")

    > 🎥 Haz clic en la imagen de arriba para ver un video: Blocks world con SHRDLU

---
## 1974 - 1980: "Invierno de la IA"

A mediados de los años 70, se hizo evidente que la complejidad de crear 'máquinas inteligentes' había sido subestimada y que su promesa, dada la potencia computacional disponible, había sido exagerada. Los fondos se agotaron y la confianza en el campo disminuyó. Algunos problemas que afectaron la confianza incluyeron:
---
- **Limitaciones**. La potencia computacional era demasiado limitada.
- **Explosión combinatoria**. La cantidad de parámetros necesarios para entrenar creció exponencialmente a medida que se pedía más a las computadoras, sin una evolución paralela de la potencia y capacidad computacional.
- **Escasez de datos**. Había una escasez de datos que dificultaba el proceso de probar, desarrollar y refinar algoritmos.
- **¿Estamos haciendo las preguntas correctas?**. Las mismas preguntas que se estaban planteando comenzaron a ser cuestionadas. Los investigadores comenzaron a recibir críticas sobre sus enfoques:
  - Los tests de Turing fueron cuestionados mediante, entre otras ideas, la 'teoría de la habitación china', que postulaba que, "programar una computadora digital puede hacer que parezca entender el lenguaje pero no podría producir una comprensión real". ([fuente](https://plato.stanford.edu/entries/chinese-room/))
  - Se cuestionó la ética de introducir inteligencias artificiales como el "terapeuta" ELIZA en la sociedad.

---

Al mismo tiempo, comenzaron a formarse varias escuelas de pensamiento sobre IA. Se estableció una dicotomía entre las prácticas de ["IA desordenada" vs. "IA ordenada"](https://wikipedia.org/wiki/Neats_and_scruffies). Los laboratorios _desordenados_ ajustaban programas durante horas hasta obtener los resultados deseados. Los laboratorios _ordenados_ "se enfocaban en la lógica y la resolución formal de problemas". ELIZA y SHRDLU eran sistemas _desordenados_ bien conocidos. En la década de 1980, a medida que surgió la demanda de hacer que los sistemas de aprendizaje automático fueran reproducibles, el enfoque _ordenado_ gradualmente tomó la delantera, ya que sus resultados son más explicables.

---
## Sistemas expertos en los años 80

A medida que el campo creció, su beneficio para los negocios se hizo más claro, y en la década de 1980 también lo hizo la proliferación de 'sistemas expertos'. "Los sistemas expertos estuvieron entre las primeras formas verdaderamente exitosas de software de inteligencia artificial (IA)." ([fuente](https://wikipedia.org/wiki/Expert_system)).

Este tipo de sistema es en realidad _híbrido_, compuesto parcialmente por un motor de reglas que define los requisitos empresariales y un motor de inferencia que aprovecha el sistema de reglas para deducir nuevos hechos.

Esta era también vio una creciente atención hacia las redes neuronales.

---
## 1987 - 1993: Enfriamiento de la IA

La proliferación de hardware especializado para sistemas expertos tuvo el desafortunado efecto de volverse demasiado especializado. El auge de las computadoras personales también compitió con estos sistemas grandes, especializados y centralizados. La democratización de la informática había comenzado, y eventualmente allanó el camino para la explosión moderna de big data.

---
## 1993 - 2011

Esta época marcó una nueva era para el aprendizaje automático y la IA, permitiendo resolver algunos de los problemas causados anteriormente por la falta de datos y potencia computacional. La cantidad de datos comenzó a aumentar rápidamente y a estar más ampliamente disponible, para bien y para mal, especialmente con la llegada del smartphone alrededor de 2007. La potencia computacional se expandió exponencialmente, y los algoritmos evolucionaron junto con ella. El campo comenzó a ganar madurez a medida que los días desenfrenados del pasado comenzaron a cristalizarse en una verdadera disciplina.

---
## Hoy

Hoy en día, el aprendizaje automático y la IA tocan casi todas las partes de nuestras vidas. Esta era exige una comprensión cuidadosa de los riesgos y los efectos potenciales de estos algoritmos en la vida humana. Como ha afirmado Brad Smith de Microsoft, "La tecnología de la información plantea cuestiones que van al corazón de las protecciones fundamentales de los derechos humanos, como la privacidad y la libertad de expresión. Estas cuestiones aumentan la responsabilidad de las empresas tecnológicas que crean estos productos. En nuestra opinión, también exigen una regulación gubernamental reflexiva y el desarrollo de normas sobre usos aceptables" ([fuente](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Queda por ver qué depara el futuro, pero es importante comprender estos sistemas informáticos y el software y los algoritmos que ejecutan. Esperamos que este plan de estudios te ayude a obtener una mejor comprensión para que puedas decidir por ti mismo.

[![La historia del aprendizaje profundo](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "La historia del aprendizaje profundo")
> 🎥 Haz clic en la imagen de arriba para ver un video: Yann LeCun habla sobre la historia del aprendizaje profundo en esta conferencia

---
## 🚀Desafío

Investiga uno de estos momentos históricos y aprende más sobre las personas detrás de ellos. Hay personajes fascinantes, y ningún descubrimiento científico se creó jamás en un vacío cultural. ¿Qué descubres?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

---
## Revisión y autoestudio

Aquí tienes elementos para ver y escuchar:

[Este podcast donde Amy Boyd analiza la evolución de la IA](http://runasradio.com/Shows/Show/739)

[![La historia de la IA por Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "La historia de la IA por Amy Boyd")

---

## Tarea

[Crear una línea de tiempo](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-04T22:20:36+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "es"
}
-->
# Construyendo soluciones de aprendizaje automático con IA responsable

![Resumen de IA responsable en aprendizaje automático en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-fairness.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

En este currículo, comenzarás a descubrir cómo el aprendizaje automático puede y está impactando nuestras vidas cotidianas. Incluso ahora, los sistemas y modelos están involucrados en tareas de toma de decisiones diarias, como diagnósticos médicos, aprobaciones de préstamos o detección de fraudes. Por lo tanto, es importante que estos modelos funcionen bien para proporcionar resultados confiables. Al igual que cualquier aplicación de software, los sistemas de IA pueden no cumplir con las expectativas o tener un resultado indeseable. Es por eso que es esencial poder entender y explicar el comportamiento de un modelo de IA.

Imagina lo que puede suceder cuando los datos que utilizas para construir estos modelos carecen de ciertos grupos demográficos, como raza, género, visión política, religión, o representan desproporcionadamente dichos grupos. ¿Qué pasa cuando la salida del modelo se interpreta para favorecer a algún grupo demográfico? ¿Cuál es la consecuencia para la aplicación? Además, ¿qué sucede cuando el modelo tiene un resultado adverso y es perjudicial para las personas? ¿Quién es responsable del comportamiento de los sistemas de IA? Estas son algunas preguntas que exploraremos en este currículo.

En esta lección, aprenderás a:

- Concienciarte sobre la importancia de la equidad en el aprendizaje automático y los daños relacionados con la falta de equidad.
- Familiarizarte con la práctica de explorar valores atípicos y escenarios inusuales para garantizar confiabilidad y seguridad.
- Comprender la necesidad de empoderar a todos mediante el diseño de sistemas inclusivos.
- Explorar lo vital que es proteger la privacidad y seguridad de los datos y las personas.
- Ver la importancia de tener un enfoque de caja de cristal para explicar el comportamiento de los modelos de IA.
- Ser consciente de cómo la responsabilidad es esencial para generar confianza en los sistemas de IA.

## Prerrequisito

Como prerrequisito, toma el "Camino de Aprendizaje de Principios de IA Responsable" y mira el siguiente video sobre el tema:

Aprende más sobre IA Responsable siguiendo este [Camino de Aprendizaje](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Enfoque de Microsoft hacia la IA Responsable](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Enfoque de Microsoft hacia la IA Responsable")

> 🎥 Haz clic en la imagen de arriba para ver el video: Enfoque de Microsoft hacia la IA Responsable

## Equidad

Los sistemas de IA deben tratar a todos de manera justa y evitar afectar a grupos similares de personas de diferentes maneras. Por ejemplo, cuando los sistemas de IA proporcionan orientación sobre tratamientos médicos, solicitudes de préstamos o empleo, deben hacer las mismas recomendaciones a todos con síntomas similares, circunstancias financieras o calificaciones profesionales. Cada uno de nosotros, como humanos, lleva consigo sesgos heredados que afectan nuestras decisiones y acciones. Estos sesgos pueden ser evidentes en los datos que usamos para entrenar sistemas de IA. A veces, esta manipulación ocurre de manera no intencional. A menudo es difícil saber conscientemente cuándo estás introduciendo sesgos en los datos.

**“Injusticia”** abarca impactos negativos, o “daños”, para un grupo de personas, como aquellos definidos en términos de raza, género, edad o estado de discapacidad. Los principales daños relacionados con la equidad pueden clasificarse como:

- **Asignación**, si, por ejemplo, se favorece un género o etnia sobre otro.
- **Calidad del servicio**. Si entrenas los datos para un escenario específico pero la realidad es mucho más compleja, esto lleva a un servicio de bajo rendimiento. Por ejemplo, un dispensador de jabón que no parece ser capaz de detectar personas con piel oscura. [Referencia](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigración**. Criticar y etiquetar algo o alguien de manera injusta. Por ejemplo, una tecnología de etiquetado de imágenes etiquetó erróneamente imágenes de personas de piel oscura como gorilas.
- **Sobre- o sub-representación**. La idea de que un cierto grupo no se ve en una determinada profesión, y cualquier servicio o función que siga promoviendo eso está contribuyendo al daño.
- **Estereotipos**. Asociar un grupo dado con atributos preasignados. Por ejemplo, un sistema de traducción entre inglés y turco puede tener inexactitudes debido a palabras con asociaciones estereotípicas de género.

![traducción al turco](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> traducción al turco

![traducción de vuelta al inglés](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> traducción de vuelta al inglés

Al diseñar y probar sistemas de IA, debemos asegurarnos de que la IA sea justa y no esté programada para tomar decisiones sesgadas o discriminatorias, las cuales también están prohibidas para los seres humanos. Garantizar la equidad en la IA y el aprendizaje automático sigue siendo un desafío sociotécnico complejo.

### Confiabilidad y seguridad

Para generar confianza, los sistemas de IA deben ser confiables, seguros y consistentes bajo condiciones normales e inesperadas. Es importante saber cómo se comportarán los sistemas de IA en una variedad de situaciones, especialmente cuando son valores atípicos. Al construir soluciones de IA, se necesita un enfoque sustancial en cómo manejar una amplia variedad de circunstancias que las soluciones de IA podrían encontrar. Por ejemplo, un automóvil autónomo debe priorizar la seguridad de las personas. Como resultado, la IA que impulsa el automóvil debe considerar todos los posibles escenarios que el automóvil podría enfrentar, como la noche, tormentas eléctricas o ventiscas, niños cruzando la calle, mascotas, construcciones en la carretera, etc. Qué tan bien un sistema de IA puede manejar una amplia gama de condiciones de manera confiable y segura refleja el nivel de anticipación que el científico de datos o desarrollador de IA consideró durante el diseño o prueba del sistema.

> [🎥 Haz clic aquí para ver un video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusión

Los sistemas de IA deben diseñarse para involucrar y empoderar a todos. Al diseñar e implementar sistemas de IA, los científicos de datos y desarrolladores de IA identifican y abordan posibles barreras en el sistema que podrían excluir a las personas de manera no intencional. Por ejemplo, hay 1,000 millones de personas con discapacidades en todo el mundo. Con el avance de la IA, pueden acceder a una amplia gama de información y oportunidades más fácilmente en su vida diaria. Al abordar las barreras, se crean oportunidades para innovar y desarrollar productos de IA con mejores experiencias que beneficien a todos.

> [🎥 Haz clic aquí para ver un video: inclusión en IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Seguridad y privacidad

Los sistemas de IA deben ser seguros y respetar la privacidad de las personas. Las personas tienen menos confianza en sistemas que ponen en riesgo su privacidad, información o vidas. Al entrenar modelos de aprendizaje automático, dependemos de los datos para producir los mejores resultados. Al hacerlo, se debe considerar el origen de los datos y su integridad. Por ejemplo, ¿los datos fueron enviados por el usuario o estaban disponibles públicamente? Luego, al trabajar con los datos, es crucial desarrollar sistemas de IA que puedan proteger información confidencial y resistir ataques. A medida que la IA se vuelve más prevalente, proteger la privacidad y asegurar información personal y empresarial importante se está volviendo más crítico y complejo. Los problemas de privacidad y seguridad de los datos requieren especial atención en la IA porque el acceso a los datos es esencial para que los sistemas de IA hagan predicciones y decisiones precisas e informadas sobre las personas.

> [🎥 Haz clic aquí para ver un video: seguridad en IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como industria, hemos logrado avances significativos en privacidad y seguridad, impulsados significativamente por regulaciones como el GDPR (Reglamento General de Protección de Datos).
- Sin embargo, con los sistemas de IA debemos reconocer la tensión entre la necesidad de más datos personales para hacer los sistemas más efectivos y la privacidad.
- Al igual que con el nacimiento de las computadoras conectadas a internet, también estamos viendo un gran aumento en el número de problemas de seguridad relacionados con la IA.
- Al mismo tiempo, hemos visto que la IA se utiliza para mejorar la seguridad. Por ejemplo, la mayoría de los escáneres antivirus modernos están impulsados por heurísticas de IA.
- Necesitamos asegurarnos de que nuestros procesos de Ciencia de Datos se integren armoniosamente con las últimas prácticas de privacidad y seguridad.

### Transparencia

Los sistemas de IA deben ser comprensibles. Una parte crucial de la transparencia es explicar el comportamiento de los sistemas de IA y sus componentes. Mejorar la comprensión de los sistemas de IA requiere que las partes interesadas comprendan cómo y por qué funcionan para que puedan identificar posibles problemas de rendimiento, preocupaciones de seguridad y privacidad, sesgos, prácticas excluyentes o resultados no deseados. También creemos que quienes usan sistemas de IA deben ser honestos y transparentes sobre cuándo, por qué y cómo eligen implementarlos, así como las limitaciones de los sistemas que utilizan. Por ejemplo, si un banco utiliza un sistema de IA para apoyar sus decisiones de préstamos al consumidor, es importante examinar los resultados y entender qué datos influyen en las recomendaciones del sistema. Los gobiernos están comenzando a regular la IA en diversas industrias, por lo que los científicos de datos y las organizaciones deben explicar si un sistema de IA cumple con los requisitos regulatorios, especialmente cuando hay un resultado no deseado.

> [🎥 Haz clic aquí para ver un video: transparencia en IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Debido a que los sistemas de IA son tan complejos, es difícil entender cómo funcionan e interpretar los resultados.
- Esta falta de comprensión afecta la forma en que se gestionan, operacionalizan y documentan estos sistemas.
- Más importante aún, esta falta de comprensión afecta las decisiones tomadas utilizando los resultados que producen estos sistemas.

### Responsabilidad

Las personas que diseñan y despliegan sistemas de IA deben ser responsables de cómo operan sus sistemas. La necesidad de responsabilidad es particularmente crucial con tecnologías de uso sensible como el reconocimiento facial. Recientemente, ha habido una creciente demanda de tecnología de reconocimiento facial, especialmente por parte de organizaciones de aplicación de la ley que ven el potencial de la tecnología en usos como encontrar niños desaparecidos. Sin embargo, estas tecnologías podrían ser utilizadas por un gobierno para poner en riesgo las libertades fundamentales de sus ciudadanos, por ejemplo, habilitando la vigilancia continua de individuos específicos. Por lo tanto, los científicos de datos y las organizaciones deben ser responsables de cómo su sistema de IA impacta a las personas o la sociedad.

[![Investigador líder en IA advierte sobre vigilancia masiva a través del reconocimiento facial](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Enfoque de Microsoft hacia la IA Responsable")

> 🎥 Haz clic en la imagen de arriba para ver el video: Advertencias sobre vigilancia masiva a través del reconocimiento facial

En última instancia, una de las mayores preguntas para nuestra generación, como la primera generación que está llevando la IA a la sociedad, es cómo garantizar que las computadoras sigan siendo responsables ante las personas y cómo garantizar que las personas que diseñan computadoras sean responsables ante todos los demás.

## Evaluación de impacto

Antes de entrenar un modelo de aprendizaje automático, es importante realizar una evaluación de impacto para entender el propósito del sistema de IA; cuál es el uso previsto; dónde se desplegará; y quién interactuará con el sistema. Esto es útil para los revisores o evaluadores del sistema para saber qué factores considerar al identificar riesgos potenciales y consecuencias esperadas.

Las siguientes son áreas de enfoque al realizar una evaluación de impacto:

* **Impacto adverso en individuos**. Ser consciente de cualquier restricción o requisito, uso no compatible o cualquier limitación conocida que obstaculice el rendimiento del sistema es vital para garantizar que el sistema no se utilice de manera que pueda causar daño a las personas.
* **Requisitos de datos**. Comprender cómo y dónde el sistema utilizará datos permite a los revisores explorar cualquier requisito de datos que debas tener en cuenta (por ejemplo, regulaciones de datos como GDPR o HIPAA). Además, examina si la fuente o cantidad de datos es suficiente para el entrenamiento.
* **Resumen del impacto**. Reúne una lista de posibles daños que podrían surgir del uso del sistema. A lo largo del ciclo de vida del aprendizaje automático, revisa si los problemas identificados se han mitigado o abordado.
* **Metas aplicables** para cada uno de los seis principios fundamentales. Evalúa si las metas de cada principio se cumplen y si hay alguna brecha.

## Depuración con IA responsable

Al igual que depurar una aplicación de software, depurar un sistema de IA es un proceso necesario para identificar y resolver problemas en el sistema. Hay muchos factores que pueden afectar que un modelo no funcione como se espera o de manera responsable. La mayoría de las métricas tradicionales de rendimiento de modelos son agregados cuantitativos del rendimiento de un modelo, lo cual no es suficiente para analizar cómo un modelo viola los principios de IA responsable. Además, un modelo de aprendizaje automático es una caja negra que dificulta entender qué impulsa su resultado o proporcionar explicaciones cuando comete un error. Más adelante en este curso, aprenderemos cómo usar el panel de IA Responsable para ayudar a depurar sistemas de IA. El panel proporciona una herramienta integral para que los científicos de datos y desarrolladores de IA realicen:

* **Análisis de errores**. Para identificar la distribución de errores del modelo que puede afectar la equidad o confiabilidad del sistema.
* **Visión general del modelo**. Para descubrir dónde hay disparidades en el rendimiento del modelo entre cohortes de datos.
* **Análisis de datos**. Para entender la distribución de datos e identificar cualquier sesgo potencial en los datos que podría generar problemas de equidad, inclusión y confiabilidad.
* **Interpretabilidad del modelo**. Para entender qué afecta o influye en las predicciones del modelo. Esto ayuda a explicar el comportamiento del modelo, lo cual es importante para la transparencia y la responsabilidad.

## 🚀 Desafío

Para prevenir daños desde el principio, deberíamos:

- contar con diversidad de antecedentes y perspectivas entre las personas que trabajan en los sistemas
- invertir en conjuntos de datos que reflejen la diversidad de nuestra sociedad
- desarrollar mejores métodos a lo largo del ciclo de vida del aprendizaje automático para detectar y corregir problemas de IA responsable cuando ocurran

Piensa en escenarios de la vida real donde la falta de confianza en un modelo sea evidente en su construcción y uso. ¿Qué más deberíamos considerar?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

En esta lección, has aprendido algunos conceptos básicos sobre la equidad y la falta de equidad en el aprendizaje automático.
Mira este taller para profundizar en los temas:

- En busca de una IA responsable: Llevando los principios a la práctica por Besmira Nushi, Mehrnoosh Sameki y Amit Sharma

[![Responsible AI Toolbox: Un marco de código abierto para construir IA responsable](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Un marco de código abierto para construir IA responsable")

> 🎥 Haz clic en la imagen de arriba para ver el video: RAI Toolbox: Un marco de código abierto para construir IA responsable por Besmira Nushi, Mehrnoosh Sameki y Amit Sharma

Además, lee:

- Centro de recursos de RAI de Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de investigación FATE de Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repositorio de GitHub de Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Lee sobre las herramientas de Azure Machine Learning para garantizar la equidad:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tarea

[Explora RAI Toolbox](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-04T22:21:15+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "es"
}
-->
# Técnicas de Aprendizaje Automático

El proceso de construir, usar y mantener modelos de aprendizaje automático y los datos que utilizan es muy diferente de muchos otros flujos de trabajo de desarrollo. En esta lección, desmitificaremos el proceso y describiremos las principales técnicas que necesitas conocer. Tú:

- Comprenderás los procesos que sustentan el aprendizaje automático a un nivel general.
- Explorarás conceptos básicos como 'modelos', 'predicciones' y 'datos de entrenamiento'.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

[![ML para principiantes - Técnicas de Aprendizaje Automático](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para principiantes - Técnicas de Aprendizaje Automático")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre esta lección.

## Introducción

A un nivel general, el arte de crear procesos de aprendizaje automático (ML) se compone de varios pasos:

1. **Decidir la pregunta**. La mayoría de los procesos de ML comienzan formulando una pregunta que no puede ser respondida mediante un programa condicional simple o un motor basado en reglas. Estas preguntas suelen girar en torno a predicciones basadas en una colección de datos.
2. **Recopilar y preparar datos**. Para poder responder a tu pregunta, necesitas datos. La calidad y, a veces, la cantidad de tus datos determinarán qué tan bien puedes responder a tu pregunta inicial. Visualizar los datos es un aspecto importante de esta fase. Esta fase también incluye dividir los datos en un grupo de entrenamiento y prueba para construir un modelo.
3. **Elegir un método de entrenamiento**. Dependiendo de tu pregunta y la naturaleza de tus datos, necesitas elegir cómo deseas entrenar un modelo para reflejar mejor tus datos y hacer predicciones precisas. Esta es la parte de tu proceso de ML que requiere experiencia específica y, a menudo, una cantidad considerable de experimentación.
4. **Entrenar el modelo**. Usando tus datos de entrenamiento, utilizarás varios algoritmos para entrenar un modelo que reconozca patrones en los datos. El modelo puede aprovechar pesos internos que se ajustan para privilegiar ciertas partes de los datos sobre otras y construir un mejor modelo.
5. **Evaluar el modelo**. Utilizas datos nunca antes vistos (tus datos de prueba) de tu conjunto recopilado para ver cómo está funcionando el modelo.
6. **Ajuste de parámetros**. Basándote en el rendimiento de tu modelo, puedes repetir el proceso utilizando diferentes parámetros o variables que controlan el comportamiento de los algoritmos utilizados para entrenar el modelo.
7. **Predecir**. Usa nuevas entradas para probar la precisión de tu modelo.

## Qué pregunta hacer

Las computadoras son particularmente hábiles para descubrir patrones ocultos en los datos. Esta utilidad es muy útil para los investigadores que tienen preguntas sobre un dominio dado que no pueden ser respondidas fácilmente creando un motor basado en reglas condicionales. Dado un trabajo actuarial, por ejemplo, un científico de datos podría construir reglas manuales sobre la mortalidad de fumadores frente a no fumadores.

Sin embargo, cuando se introducen muchas otras variables en la ecuación, un modelo de ML podría resultar más eficiente para predecir tasas de mortalidad futuras basándose en historiales de salud pasados. Un ejemplo más alegre podría ser hacer predicciones meteorológicas para el mes de abril en una ubicación dada basándose en datos que incluyen latitud, longitud, cambio climático, proximidad al océano, patrones de la corriente en chorro y más.

✅ Este [conjunto de diapositivas](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos ofrece una perspectiva histórica sobre el uso de ML en el análisis del clima.  

## Tareas previas a la construcción

Antes de comenzar a construir tu modelo, hay varias tareas que necesitas completar. Para probar tu pregunta y formular una hipótesis basada en las predicciones de un modelo, necesitas identificar y configurar varios elementos.

### Datos

Para poder responder a tu pregunta con algún tipo de certeza, necesitas una buena cantidad de datos del tipo correcto. Hay dos cosas que necesitas hacer en este punto:

- **Recopilar datos**. Teniendo en cuenta la lección anterior sobre equidad en el análisis de datos, recopila tus datos con cuidado. Sé consciente de las fuentes de estos datos, cualquier sesgo inherente que puedan tener y documenta su origen.
- **Preparar datos**. Hay varios pasos en el proceso de preparación de datos. Es posible que necesites compilar datos y normalizarlos si provienen de fuentes diversas. Puedes mejorar la calidad y cantidad de los datos mediante varios métodos, como convertir cadenas en números (como hacemos en [Clustering](../../5-Clustering/1-Visualize/README.md)). También puedes generar nuevos datos basados en los originales (como hacemos en [Clasificación](../../4-Classification/1-Introduction/README.md)). Puedes limpiar y editar los datos (como haremos antes de la lección de [Aplicación Web](../../3-Web-App/README.md)). Finalmente, también podrías necesitar aleatorizarlos y mezclarlos, dependiendo de tus técnicas de entrenamiento.

✅ Después de recopilar y procesar tus datos, tómate un momento para ver si su forma te permitirá abordar tu pregunta. Puede ser que los datos no funcionen bien en tu tarea dada, como descubrimos en nuestras lecciones de [Clustering](../../5-Clustering/1-Visualize/README.md).

### Características y Objetivo

Una [característica](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) es una propiedad medible de tus datos. En muchos conjuntos de datos se expresa como un encabezado de columna como 'fecha', 'tamaño' o 'color'. Tu variable de característica, usualmente representada como `X` en el código, representa la variable de entrada que se usará para entrenar el modelo.

Un objetivo es aquello que estás tratando de predecir. El objetivo, usualmente representado como `y` en el código, representa la respuesta a la pregunta que estás tratando de hacer a tus datos: en diciembre, ¿qué **color** de calabazas será el más barato? En San Francisco, ¿qué vecindarios tendrán el mejor **precio** inmobiliario? A veces el objetivo también se denomina atributo de etiqueta.

### Selección de tu variable de característica

🎓 **Selección de Características y Extracción de Características** ¿Cómo sabes qué variable elegir al construir un modelo? Probablemente pasarás por un proceso de selección de características o extracción de características para elegir las variables correctas para el modelo más eficiente. Sin embargo, no son lo mismo: "La extracción de características crea nuevas características a partir de funciones de las características originales, mientras que la selección de características devuelve un subconjunto de las características." ([fuente](https://wikipedia.org/wiki/Feature_selection))

### Visualiza tus datos

Un aspecto importante del conjunto de herramientas del científico de datos es el poder de visualizar datos utilizando varias bibliotecas excelentes como Seaborn o MatPlotLib. Representar tus datos visualmente podría permitirte descubrir correlaciones ocultas que puedes aprovechar. Tus visualizaciones también podrían ayudarte a descubrir sesgos o datos desequilibrados (como descubrimos en [Clasificación](../../4-Classification/2-Classifiers-1/README.md)).

### Divide tu conjunto de datos

Antes de entrenar, necesitas dividir tu conjunto de datos en dos o más partes de tamaño desigual que aún representen bien los datos.

- **Entrenamiento**. Esta parte del conjunto de datos se ajusta a tu modelo para entrenarlo. Este conjunto constituye la mayoría del conjunto de datos original.
- **Prueba**. Un conjunto de prueba es un grupo independiente de datos, a menudo recopilado del conjunto original, que utilizas para confirmar el rendimiento del modelo construido.
- **Validación**. Un conjunto de validación es un grupo independiente más pequeño de ejemplos que utilizas para ajustar los hiperparámetros o la arquitectura del modelo para mejorarlo. Dependiendo del tamaño de tus datos y la pregunta que estás haciendo, es posible que no necesites construir este tercer conjunto (como señalamos en [Pronóstico de Series Temporales](../../7-TimeSeries/1-Introduction/README.md)).

## Construcción de un modelo

Usando tus datos de entrenamiento, tu objetivo es construir un modelo, o una representación estadística de tus datos, utilizando varios algoritmos para **entrenarlo**. Entrenar un modelo lo expone a datos y le permite hacer suposiciones sobre patrones percibidos que descubre, valida y acepta o rechaza.

### Decidir un método de entrenamiento

Dependiendo de tu pregunta y la naturaleza de tus datos, elegirás un método para entrenarlo. Al recorrer la [documentación de Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos en este curso - puedes explorar muchas formas de entrenar un modelo. Dependiendo de tu experiencia, es posible que tengas que probar varios métodos diferentes para construir el mejor modelo. Es probable que pases por un proceso en el que los científicos de datos evalúan el rendimiento de un modelo alimentándolo con datos no vistos, verificando su precisión, sesgo y otros problemas que degradan la calidad, y seleccionando el método de entrenamiento más apropiado para la tarea en cuestión.

### Entrenar un modelo

Con tus datos de entrenamiento, estás listo para 'ajustarlo' y crear un modelo. Notarás que en muchas bibliotecas de ML encontrarás el código 'model.fit' - es en este momento que envías tu variable de característica como un arreglo de valores (usualmente 'X') y una variable objetivo (usualmente 'y').

### Evaluar el modelo

Una vez que el proceso de entrenamiento esté completo (puede tomar muchas iteraciones, o 'épocas', para entrenar un modelo grande), podrás evaluar la calidad del modelo utilizando datos de prueba para medir su rendimiento. Estos datos son un subconjunto de los datos originales que el modelo no ha analizado previamente. Puedes imprimir una tabla de métricas sobre la calidad de tu modelo.

🎓 **Ajuste del modelo**

En el contexto del aprendizaje automático, el ajuste del modelo se refiere a la precisión de la función subyacente del modelo mientras intenta analizar datos con los que no está familiarizado.

🎓 **Subajuste** y **sobreajuste** son problemas comunes que degradan la calidad del modelo, ya que el modelo se ajusta demasiado poco o demasiado bien. Esto hace que el modelo haga predicciones demasiado alineadas o demasiado poco alineadas con sus datos de entrenamiento. Un modelo sobreajustado predice los datos de entrenamiento demasiado bien porque ha aprendido demasiado bien los detalles y el ruido de los datos. Un modelo subajustado no es preciso ya que no puede analizar con precisión ni sus datos de entrenamiento ni los datos que aún no ha 'visto'.

![modelo sobreajustado](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografía por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parámetros

Una vez que tu entrenamiento inicial esté completo, observa la calidad del modelo y considera mejorarlo ajustando sus 'hiperparámetros'. Lee más sobre el proceso [en la documentación](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predicción

Este es el momento en el que puedes usar datos completamente nuevos para probar la precisión de tu modelo. En un entorno de ML 'aplicado', donde estás construyendo activos web para usar el modelo en producción, este proceso podría implicar recopilar la entrada del usuario (por ejemplo, presionar un botón) para establecer una variable y enviarla al modelo para inferencia o evaluación.

En estas lecciones, descubrirás cómo usar estos pasos para preparar, construir, probar, evaluar y predecir: todos los gestos de un científico de datos y más, mientras avanzas en tu camino para convertirte en un ingeniero de ML 'full stack'.

---

## 🚀Desafío

Dibuja un diagrama de flujo que refleje los pasos de un practicante de ML. ¿Dónde te ves ahora en el proceso? ¿Dónde predices que encontrarás dificultades? ¿Qué te parece fácil?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

Busca en línea entrevistas con científicos de datos que hablen sobre su trabajo diario. Aquí tienes [una](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarea

[Entrevista a un científico de datos](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---


# Regresión

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-03T22:15:28+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "es"
}
-->
# Modelos de regresión para aprendizaje automático
## Tema regional: Modelos de regresión para precios de calabazas en América del Norte 🎃

En América del Norte, las calabazas suelen tallarse con caras aterradoras para Halloween. ¡Descubramos más sobre estos fascinantes vegetales!

![jack-o-lanterns](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/jack-o-lanterns.181c661a9212457d.webp)
> Foto por <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> en <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Lo que aprenderás

[![Introducción a la regresión](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Video de introducción a la regresión - ¡Haz clic para verlo!")
> 🎥 Haz clic en la imagen de arriba para ver un video introductorio rápido de esta lección.

Las lecciones en esta sección cubren los tipos de regresión en el contexto del aprendizaje automático. Los modelos de regresión pueden ayudar a determinar la _relación_ entre variables. Este tipo de modelo puede predecir valores como longitud, temperatura o edad, revelando así relaciones entre variables mientras analiza puntos de datos.

En esta serie de lecciones, descubrirás las diferencias entre regresión lineal y logística, y cuándo deberías preferir una sobre la otra.

[![ML para principiantes - Introducción a los modelos de regresión para aprendizaje automático](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML para principiantes - Introducción a los modelos de regresión para aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un breve video que introduce los modelos de regresión.

En este grupo de lecciones, te prepararás para comenzar tareas de aprendizaje automático, incluyendo la configuración de Visual Studio Code para gestionar notebooks, el entorno común para los científicos de datos. Descubrirás Scikit-learn, una biblioteca para aprendizaje automático, y construirás tus primeros modelos, enfocándote en modelos de regresión en este capítulo.

> Hay herramientas útiles de bajo código que pueden ayudarte a aprender sobre el trabajo con modelos de regresión. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### Lecciones

1. [Herramientas del oficio](1-Tools/README.md)
2. [Gestión de datos](2-Data/README.md)
3. [Regresión lineal y polinómica](3-Linear/README.md)
4. [Regresión logística](4-Logistic/README.md)

---
### Créditos

"ML con regresión" fue escrito con ♥️ por [Jen Looper](https://twitter.com/jenlooper)

♥️ Los colaboradores de los cuestionarios incluyen: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) y [Ornella Altunyan](https://twitter.com/ornelladotcom)

El conjunto de datos de calabazas es sugerido por [este proyecto en Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) y sus datos provienen de los [Informes estándar de mercados terminales de cultivos especializados](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuidos por el Departamento de Agricultura de los Estados Unidos. Hemos añadido algunos puntos sobre el color según la variedad para normalizar la distribución. Estos datos están en el dominio público.

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-04T22:13:49+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "es"
}
-->
# Comienza con Python y Scikit-learn para modelos de regresión

![Resumen de regresiones en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-regression.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introducción

En estas cuatro lecciones, descubrirás cómo construir modelos de regresión. Hablaremos sobre para qué sirven en breve. Pero antes de hacer nada, ¡asegúrate de tener las herramientas adecuadas para comenzar el proceso!

En esta lección, aprenderás a:

- Configurar tu computadora para tareas locales de aprendizaje automático.
- Trabajar con Jupyter notebooks.
- Usar Scikit-learn, incluyendo su instalación.
- Explorar la regresión lineal con un ejercicio práctico.

## Instalaciones y configuraciones

[![ML para principiantes - Configura tus herramientas para construir modelos de aprendizaje automático](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para principiantes - Configura tus herramientas para construir modelos de aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un video corto sobre cómo configurar tu computadora para ML.

1. **Instalar Python**. Asegúrate de que [Python](https://www.python.org/downloads/) esté instalado en tu computadora. Usarás Python para muchas tareas de ciencia de datos y aprendizaje automático. La mayoría de los sistemas informáticos ya incluyen una instalación de Python. También hay disponibles [Paquetes de Codificación de Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) útiles para facilitar la configuración a algunos usuarios.

   Sin embargo, algunos usos de Python requieren una versión específica del software, mientras que otros requieren una versión diferente. Por esta razón, es útil trabajar dentro de un [entorno virtual](https://docs.python.org/3/library/venv.html).

2. **Instalar Visual Studio Code**. Asegúrate de tener Visual Studio Code instalado en tu computadora. Sigue estas instrucciones para [instalar Visual Studio Code](https://code.visualstudio.com/) para la instalación básica. Vas a usar Python en Visual Studio Code en este curso, por lo que podrías querer repasar cómo [configurar Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para el desarrollo en Python.

   > Familiarízate con Python trabajando en esta colección de [módulos de aprendizaje](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configura Python con Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configura Python con Visual Studio Code")
   >
   > 🎥 Haz clic en la imagen de arriba para ver un video: usando Python dentro de VS Code.

3. **Instalar Scikit-learn**, siguiendo [estas instrucciones](https://scikit-learn.org/stable/install.html). Dado que necesitas asegurarte de usar Python 3, se recomienda que utilices un entorno virtual. Nota: si estás instalando esta biblioteca en una Mac M1, hay instrucciones especiales en la página enlazada arriba.

4. **Instalar Jupyter Notebook**. Necesitarás [instalar el paquete Jupyter](https://pypi.org/project/jupyter/).

## Tu entorno de autoría de ML

Vas a usar **notebooks** para desarrollar tu código en Python y crear modelos de aprendizaje automático. Este tipo de archivo es una herramienta común para los científicos de datos y se identifican por su sufijo o extensión `.ipynb`.

Los notebooks son un entorno interactivo que permite al desarrollador tanto codificar como agregar notas y escribir documentación alrededor del código, lo cual es bastante útil para proyectos experimentales o de investigación.

[![ML para principiantes - Configura Jupyter Notebooks para comenzar a construir modelos de regresión](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para principiantes - Configura Jupyter Notebooks para comenzar a construir modelos de regresión")

> 🎥 Haz clic en la imagen de arriba para ver un video corto sobre este ejercicio.

### Ejercicio - trabajar con un notebook

En esta carpeta, encontrarás el archivo _notebook.ipynb_.

1. Abre _notebook.ipynb_ en Visual Studio Code.

   Se iniciará un servidor Jupyter con Python 3+. Encontrarás áreas del notebook que pueden ser `ejecutadas`, piezas de código. Puedes ejecutar un bloque de código seleccionando el ícono que parece un botón de reproducción.

2. Selecciona el ícono `md` y agrega un poco de markdown, y el siguiente texto **# Bienvenido a tu notebook**.

   Luego, agrega algo de código en Python.

3. Escribe **print('hello notebook')** en el bloque de código.
4. Selecciona la flecha para ejecutar el código.

   Deberías ver la declaración impresa:

    ```output
    hello notebook
    ```

![VS Code con un notebook abierto](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/1-Tools/images/notebook.jpg)

Puedes intercalar tu código con comentarios para auto-documentar el notebook.

✅ Piensa por un momento en cómo es diferente el entorno de trabajo de un desarrollador web en comparación con el de un científico de datos.

## Puesta en marcha con Scikit-learn

Ahora que Python está configurado en tu entorno local y te sientes cómodo con los notebooks de Jupyter, vamos a familiarizarnos con Scikit-learn (se pronuncia `sci` como en `science`). Scikit-learn proporciona una [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ayudarte a realizar tareas de ML.

Según su [sitio web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn es una biblioteca de aprendizaje automático de código abierto que admite aprendizaje supervisado y no supervisado. También proporciona varias herramientas para ajuste de modelos, preprocesamiento de datos, selección y evaluación de modelos, y muchas otras utilidades."

En este curso, usarás Scikit-learn y otras herramientas para construir modelos de aprendizaje automático para realizar lo que llamamos tareas de 'aprendizaje automático tradicional'. Hemos evitado deliberadamente redes neuronales y aprendizaje profundo, ya que están mejor cubiertos en nuestro próximo currículo 'AI para Principiantes'.

Scikit-learn hace que sea sencillo construir modelos y evaluarlos para su uso. Se centra principalmente en el uso de datos numéricos y contiene varios conjuntos de datos listos para usar como herramientas de aprendizaje. También incluye modelos preconstruidos para que los estudiantes los prueben. Vamos a explorar el proceso de cargar datos preempaquetados y usar un estimador para el primer modelo de ML con Scikit-learn con algunos datos básicos.

## Ejercicio - tu primer notebook con Scikit-learn

> Este tutorial fue inspirado por el [ejemplo de regresión lineal](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) en el sitio web de Scikit-learn.

[![ML para principiantes - Tu primer proyecto de regresión lineal en Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para principiantes - Tu primer proyecto de regresión lineal en Python")

> 🎥 Haz clic en la imagen de arriba para ver un video corto sobre este ejercicio.

En el archivo _notebook.ipynb_ asociado a esta lección, elimina todas las celdas presionando el ícono de 'papelera'.

En esta sección, trabajarás con un pequeño conjunto de datos sobre diabetes que está integrado en Scikit-learn para propósitos de aprendizaje. Imagina que quisieras probar un tratamiento para pacientes diabéticos. Los modelos de aprendizaje automático podrían ayudarte a determinar qué pacientes responderían mejor al tratamiento, basándote en combinaciones de variables. Incluso un modelo de regresión muy básico, cuando se visualiza, podría mostrar información sobre variables que te ayudarían a organizar tus ensayos clínicos teóricos.

✅ Hay muchos tipos de métodos de regresión, y cuál elijas depende de la respuesta que estés buscando. Si quieres predecir la altura probable de una persona dada su edad, usarías regresión lineal, ya que estás buscando un **valor numérico**. Si estás interesado en descubrir si un tipo de cocina debería considerarse vegana o no, estás buscando una **asignación de categoría**, por lo que usarías regresión logística. Aprenderás más sobre regresión logística más adelante. Piensa un poco en algunas preguntas que puedes hacer a los datos y cuál de estos métodos sería más apropiado.

Vamos a comenzar con esta tarea.

### Importar bibliotecas

Para esta tarea, importaremos algunas bibliotecas:

- **matplotlib**. Es una herramienta útil para [graficar](https://matplotlib.org/) y la usaremos para crear un gráfico de líneas.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) es una biblioteca útil para manejar datos numéricos en Python.
- **sklearn**. Esta es la biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importa algunas bibliotecas para ayudarte con tus tareas.

1. Agrega las importaciones escribiendo el siguiente código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Arriba estás importando `matplotlib`, `numpy` y estás importando `datasets`, `linear_model` y `model_selection` de `sklearn`. `model_selection` se usa para dividir datos en conjuntos de entrenamiento y prueba.

### El conjunto de datos de diabetes

El [conjunto de datos de diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) integrado incluye 442 muestras de datos sobre diabetes, con 10 variables de características, algunas de las cuales incluyen:

- age: edad en años
- bmi: índice de masa corporal
- bp: presión arterial promedio
- s1 tc: células T (un tipo de glóbulos blancos)

✅ Este conjunto de datos incluye el concepto de 'sexo' como una variable de característica importante para la investigación sobre diabetes. Muchos conjuntos de datos médicos incluyen este tipo de clasificación binaria. Piensa un poco en cómo categorizaciones como esta podrían excluir a ciertas partes de la población de los tratamientos.

Ahora, carga los datos X e y.

> 🎓 Recuerda, esto es aprendizaje supervisado, y necesitamos un objetivo 'y' nombrado.

En una nueva celda de código, carga el conjunto de datos de diabetes llamando a `load_diabetes()`. El parámetro `return_X_y=True` indica que `X` será una matriz de datos y `y` será el objetivo de regresión.

1. Agrega algunos comandos de impresión para mostrar la forma de la matriz de datos y su primer elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Lo que estás obteniendo como respuesta es una tupla. Lo que estás haciendo es asignar los dos primeros valores de la tupla a `X` e `y` respectivamente. Aprende más [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Puedes ver que estos datos tienen 442 elementos organizados en matrices de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Piensa un poco sobre la relación entre los datos y el objetivo de regresión. La regresión lineal predice relaciones entre la característica X y la variable objetivo y. ¿Puedes encontrar el [objetivo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para el conjunto de datos de diabetes en la documentación? ¿Qué está demostrando este conjunto de datos, dado ese objetivo?

2. A continuación, selecciona una porción de este conjunto de datos para graficar seleccionando la tercera columna del conjunto de datos. Puedes hacerlo usando el operador `:` para seleccionar todas las filas y luego seleccionando la tercera columna usando el índice (2). También puedes cambiar la forma de los datos para que sean una matriz 2D, como se requiere para graficar, usando `reshape(n_rows, n_columns)`. Si uno de los parámetros es -1, la dimensión correspondiente se calcula automáticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ En cualquier momento, imprime los datos para verificar su forma.

3. Ahora que tienes los datos listos para ser graficados, puedes ver si una máquina puede ayudar a determinar una división lógica entre los números en este conjunto de datos. Para hacer esto, necesitas dividir tanto los datos (X) como el objetivo (y) en conjuntos de prueba y entrenamiento. Scikit-learn tiene una forma sencilla de hacer esto; puedes dividir tus datos de prueba en un punto dado.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. ¡Ahora estás listo para entrenar tu modelo! Carga el modelo de regresión lineal y entrénalo con tus conjuntos de entrenamiento X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` es una función que verás en muchas bibliotecas de ML como TensorFlow.

5. Luego, crea una predicción usando datos de prueba, utilizando la función `predict()`. Esto se usará para dibujar la línea entre los grupos de datos.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Ahora es momento de mostrar los datos en un gráfico. Matplotlib es una herramienta muy útil para esta tarea. Crea un gráfico de dispersión de todos los datos de prueba X e y, y usa la predicción para dibujar una línea en el lugar más apropiado, entre los grupos de datos del modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![un gráfico de dispersión mostrando puntos de datos sobre diabetes](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/1-Tools/images/scatterplot.png)
✅ Piensa un poco en lo que está sucediendo aquí. Una línea recta está atravesando muchos pequeños puntos de datos, pero ¿qué está haciendo exactamente? ¿Puedes ver cómo deberías poder usar esta línea para predecir dónde debería encajar un nuevo punto de datos no visto en relación con el eje y del gráfico? Intenta poner en palabras el uso práctico de este modelo.

¡Felicidades! Has construido tu primer modelo de regresión lineal, creado una predicción con él y la has mostrado en un gráfico.

---
## 🚀Desafío

Grafica una variable diferente de este conjunto de datos. Pista: edita esta línea: `X = X[:,2]`. Dado el objetivo de este conjunto de datos, ¿qué puedes descubrir sobre la progresión de la diabetes como enfermedad?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

En este tutorial, trabajaste con regresión lineal simple, en lugar de regresión univariante o múltiple. Lee un poco sobre las diferencias entre estos métodos, o echa un vistazo a [este video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lee más sobre el concepto de regresión y piensa en qué tipo de preguntas pueden responderse con esta técnica. Toma este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para profundizar tu comprensión.

## Tarea

[Un conjunto de datos diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-04T22:14:22+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "es"
}
-->
# Construir un modelo de regresión usando Scikit-learn: preparar y visualizar datos

![Infografía de visualización de datos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/2-Data/images/data-visualization.png)

Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introducción

Ahora que tienes las herramientas necesarias para comenzar a construir modelos de aprendizaje automático con Scikit-learn, estás listo para empezar a formular preguntas sobre tus datos. Al trabajar con datos y aplicar soluciones de ML, es muy importante saber cómo hacer la pregunta correcta para desbloquear adecuadamente el potencial de tu conjunto de datos.

En esta lección, aprenderás:

- Cómo preparar tus datos para construir modelos.
- Cómo usar Matplotlib para la visualización de datos.

## Hacer la pregunta correcta sobre tus datos

La pregunta que necesitas responder determinará qué tipo de algoritmos de ML utilizarás. Y la calidad de la respuesta que obtengas dependerá en gran medida de la naturaleza de tus datos.

Echa un vistazo a los [datos](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) proporcionados para esta lección. Puedes abrir este archivo .csv en VS Code. Una revisión rápida muestra que hay espacios en blanco y una mezcla de datos de tipo cadena y numéricos. También hay una columna extraña llamada 'Package' donde los datos son una mezcla entre 'sacks', 'bins' y otros valores. De hecho, los datos están un poco desordenados.

[![ML para principiantes - Cómo analizar y limpiar un conjunto de datos](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML para principiantes - Cómo analizar y limpiar un conjunto de datos")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre cómo preparar los datos para esta lección.

De hecho, no es muy común recibir un conjunto de datos completamente listo para usar y crear un modelo de ML directamente. En esta lección, aprenderás cómo preparar un conjunto de datos sin procesar utilizando bibliotecas estándar de Python. También aprenderás varias técnicas para visualizar los datos.

## Caso de estudio: 'el mercado de calabazas'

En esta carpeta encontrarás un archivo .csv en la carpeta raíz `data` llamado [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), que incluye 1757 líneas de datos sobre el mercado de calabazas, agrupados por ciudad. Estos son datos sin procesar extraídos de los [Informes estándar de mercados terminales de cultivos especiales](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuidos por el Departamento de Agricultura de los Estados Unidos.

### Preparar los datos

Estos datos son de dominio público. Se pueden descargar en muchos archivos separados, por ciudad, desde el sitio web del USDA. Para evitar demasiados archivos separados, hemos concatenado todos los datos de las ciudades en una sola hoja de cálculo, por lo que ya hemos _preparado_ un poco los datos. Ahora, echemos un vistazo más de cerca a los datos.

### Los datos de calabazas - primeras conclusiones

¿Qué notas sobre estos datos? Ya viste que hay una mezcla de cadenas, números, espacios en blanco y valores extraños que necesitas interpretar.

¿Qué pregunta puedes hacer sobre estos datos utilizando una técnica de regresión? ¿Qué tal "Predecir el precio de una calabaza en venta durante un mes determinado"? Mirando nuevamente los datos, hay algunos cambios que necesitas hacer para crear la estructura de datos necesaria para esta tarea.

## Ejercicio - analizar los datos de calabazas

Usemos [Pandas](https://pandas.pydata.org/) (el nombre significa `Python Data Analysis`), una herramienta muy útil para dar forma a los datos, para analizar y preparar estos datos de calabazas.

### Primero, verifica si faltan fechas

Primero necesitarás tomar medidas para verificar si faltan fechas:

1. Convierte las fechas al formato de mes (estas son fechas de EE. UU., por lo que el formato es `MM/DD/YYYY`).
2. Extrae el mes a una nueva columna.

Abre el archivo _notebook.ipynb_ en Visual Studio Code e importa la hoja de cálculo en un nuevo dataframe de Pandas.

1. Usa la función `head()` para ver las primeras cinco filas.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ ¿Qué función usarías para ver las últimas cinco filas?

1. Verifica si hay datos faltantes en el dataframe actual:

    ```python
    pumpkins.isnull().sum()
    ```

    Hay datos faltantes, pero tal vez no importen para la tarea en cuestión.

1. Para que tu dataframe sea más fácil de trabajar, selecciona solo las columnas que necesitas, usando la función `loc`, que extrae del dataframe original un grupo de filas (pasadas como primer parámetro) y columnas (pasadas como segundo parámetro). La expresión `:` en el caso siguiente significa "todas las filas".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Segundo, determina el precio promedio de las calabazas

Piensa en cómo determinar el precio promedio de una calabaza en un mes determinado. ¿Qué columnas elegirías para esta tarea? Pista: necesitarás 3 columnas.

Solución: toma el promedio de las columnas `Low Price` y `High Price` para llenar la nueva columna Price, y convierte la columna Date para que solo muestre el mes. Afortunadamente, según la verificación anterior, no hay datos faltantes para fechas o precios.

1. Para calcular el promedio, agrega el siguiente código:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Siéntete libre de imprimir cualquier dato que desees verificar usando `print(month)`.

2. Ahora, copia tus datos convertidos en un nuevo dataframe de Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Al imprimir tu dataframe, verás un conjunto de datos limpio y ordenado sobre el cual puedes construir tu nuevo modelo de regresión.

### Pero espera, ¡hay algo extraño aquí!

Si miras la columna `Package`, las calabazas se venden en muchas configuraciones diferentes. Algunas se venden en medidas de '1 1/9 bushel', otras en '1/2 bushel', algunas por calabaza, otras por libra, y algunas en grandes cajas con anchos variables.

> Parece que las calabazas son muy difíciles de pesar de manera consistente.

Al profundizar en los datos originales, es interesante notar que cualquier cosa con `Unit of Sale` igual a 'EACH' o 'PER BIN' también tiene el tipo `Package` por pulgada, por bin, o 'each'. Parece que las calabazas son muy difíciles de pesar de manera consistente, así que filtremos seleccionando solo las calabazas con la cadena 'bushel' en su columna `Package`.

1. Agrega un filtro en la parte superior del archivo, debajo de la importación inicial del .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Si imprimes los datos ahora, puedes ver que solo estás obteniendo las aproximadamente 415 filas de datos que contienen calabazas por bushel.

### Pero espera, ¡hay una cosa más por hacer!

¿Notaste que la cantidad de bushel varía por fila? Necesitas normalizar los precios para mostrar el precio por bushel, así que haz algunos cálculos para estandarizarlo.

1. Agrega estas líneas después del bloque que crea el dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Según [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), el peso de un bushel depende del tipo de producto, ya que es una medida de volumen. "Un bushel de tomates, por ejemplo, se supone que pesa 56 libras... Las hojas y los vegetales ocupan más espacio con menos peso, por lo que un bushel de espinacas pesa solo 20 libras". ¡Es todo bastante complicado! No nos molestemos en hacer una conversión de bushel a libra, y en su lugar fijemos el precio por bushel. Todo este estudio sobre bushels de calabazas, sin embargo, demuestra lo importante que es entender la naturaleza de tus datos.

Ahora puedes analizar el precio por unidad basado en su medida de bushel. Si imprimes los datos una vez más, puedes ver cómo se han estandarizado.

✅ ¿Notaste que las calabazas vendidas por medio bushel son muy caras? ¿Puedes averiguar por qué? Pista: las calabazas pequeñas son mucho más caras que las grandes, probablemente porque hay muchas más por bushel, dado el espacio no utilizado que ocupa una calabaza grande y hueca para pastel.

## Estrategias de visualización

Parte del rol del científico de datos es demostrar la calidad y naturaleza de los datos con los que están trabajando. Para hacerlo, a menudo crean visualizaciones interesantes, como gráficos, diagramas y tablas, que muestran diferentes aspectos de los datos. De esta manera, pueden mostrar visualmente relaciones y brechas que de otro modo serían difíciles de descubrir.

[![ML para principiantes - Cómo visualizar datos con Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML para principiantes - Cómo visualizar datos con Matplotlib")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre cómo visualizar los datos para esta lección.

Las visualizaciones también pueden ayudar a determinar la técnica de aprendizaje automático más adecuada para los datos. Un gráfico de dispersión que parece seguir una línea, por ejemplo, indica que los datos son buenos candidatos para un ejercicio de regresión lineal.

Una biblioteca de visualización de datos que funciona bien en Jupyter notebooks es [Matplotlib](https://matplotlib.org/) (que también viste en la lección anterior).

> Obtén más experiencia con la visualización de datos en [estos tutoriales](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Ejercicio - experimentar con Matplotlib

Intenta crear algunos gráficos básicos para mostrar el nuevo dataframe que acabas de crear. ¿Qué mostraría un gráfico de líneas básico?

1. Importa Matplotlib en la parte superior del archivo, debajo de la importación de Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Vuelve a ejecutar todo el notebook para actualizar.
1. En la parte inferior del notebook, agrega una celda para graficar los datos como un cuadro:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Un gráfico de dispersión que muestra la relación entre precio y mes](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/2-Data/images/scatterplot.png)

    ¿Es este un gráfico útil? ¿Hay algo que te sorprenda?

    No es particularmente útil, ya que solo muestra tus datos como una dispersión de puntos en un mes determinado.

### Hazlo útil

Para que los gráficos muestren datos útiles, generalmente necesitas agrupar los datos de alguna manera. Intentemos crear un gráfico donde el eje y muestre los meses y los datos demuestren la distribución de los mismos.

1. Agrega una celda para crear un gráfico de barras agrupado:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Un gráfico de barras que muestra la relación entre precio y mes](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/2-Data/images/barchart.png)

    ¡Este es un gráfico de datos más útil! Parece indicar que el precio más alto de las calabazas ocurre en septiembre y octubre. ¿Cumple con tus expectativas? ¿Por qué o por qué no?

---

## 🚀Desafío

Explora los diferentes tipos de visualización que ofrece Matplotlib. ¿Qué tipos son más apropiados para problemas de regresión?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Echa un vistazo a las muchas formas de visualizar datos. Haz una lista de las diversas bibliotecas disponibles y anota cuáles son mejores para ciertos tipos de tareas, por ejemplo, visualizaciones en 2D frente a visualizaciones en 3D. ¿Qué descubres?

## Tarea

[Explorar visualización](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-04T22:11:11+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "es"
}
-->
# Construir un modelo de regresión usando Scikit-learn: cuatro formas de regresión

![Infografía de regresión lineal vs polinómica](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/3-Linear/images/linear-polynomial.png)
> Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introducción 

Hasta ahora has explorado qué es la regresión con datos de muestra obtenidos del conjunto de datos de precios de calabazas que utilizaremos a lo largo de esta lección. También lo has visualizado utilizando Matplotlib.

Ahora estás listo para profundizar más en la regresión para ML. Mientras que la visualización te permite comprender los datos, el verdadero poder del aprendizaje automático proviene del _entrenamiento de modelos_. Los modelos se entrenan con datos históricos para capturar automáticamente las dependencias de los datos, y te permiten predecir resultados para nuevos datos que el modelo no ha visto antes.

En esta lección, aprenderás más sobre dos tipos de regresión: _regresión lineal básica_ y _regresión polinómica_, junto con algunas de las matemáticas subyacentes a estas técnicas. Estos modelos nos permitirán predecir los precios de las calabazas dependiendo de diferentes datos de entrada.

[![ML para principiantes - Entendiendo la regresión lineal](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para principiantes - Entendiendo la regresión lineal")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre la regresión lineal.

> A lo largo de este plan de estudios, asumimos un conocimiento mínimo de matemáticas y buscamos hacerlo accesible para estudiantes provenientes de otros campos, así que presta atención a las notas, 🧮 llamados, diagramas y otras herramientas de aprendizaje para facilitar la comprensión.

### Prerrequisitos

A estas alturas deberías estar familiarizado con la estructura de los datos de calabazas que estamos examinando. Puedes encontrarlos precargados y preprocesados en el archivo _notebook.ipynb_ de esta lección. En el archivo, el precio de las calabazas se muestra por bushel en un nuevo marco de datos. Asegúrate de poder ejecutar estos notebooks en kernels en Visual Studio Code.

### Preparación

Como recordatorio, estás cargando estos datos para hacer preguntas sobre ellos.

- ¿Cuándo es el mejor momento para comprar calabazas? 
- ¿Qué precio puedo esperar por un paquete de calabazas miniatura?
- ¿Debería comprarlas en cestas de medio bushel o en cajas de 1 1/9 bushel?
Sigamos investigando estos datos.

En la lección anterior, creaste un marco de datos de Pandas y lo llenaste con parte del conjunto de datos original, estandarizando los precios por bushel. Sin embargo, al hacer eso, solo pudiste recopilar alrededor de 400 puntos de datos y solo para los meses de otoño.

Echa un vistazo a los datos que precargamos en el notebook que acompaña esta lección. Los datos están precargados y se ha trazado un gráfico de dispersión inicial para mostrar los datos por mes. Tal vez podamos obtener un poco más de detalle sobre la naturaleza de los datos limpiándolos más.

## Una línea de regresión lineal

Como aprendiste en la Lección 1, el objetivo de un ejercicio de regresión lineal es poder trazar una línea para:

- **Mostrar relaciones entre variables**. Mostrar la relación entre las variables.
- **Hacer predicciones**. Hacer predicciones precisas sobre dónde caería un nuevo punto de datos en relación con esa línea.

Es típico de la **Regresión de Mínimos Cuadrados** trazar este tipo de línea. El término 'mínimos cuadrados' significa que todos los puntos de datos que rodean la línea de regresión se elevan al cuadrado y luego se suman. Idealmente, esa suma final es lo más pequeña posible, porque queremos un número bajo de errores, o `mínimos cuadrados`.

Hacemos esto porque queremos modelar una línea que tenga la menor distancia acumulada de todos nuestros puntos de datos. También elevamos los términos al cuadrado antes de sumarlos porque nos interesa su magnitud más que su dirección.

> **🧮 Muéstrame las matemáticas** 
> 
> Esta línea, llamada _línea de mejor ajuste_, puede expresarse mediante [una ecuación](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` es la 'variable explicativa'. `Y` es la 'variable dependiente'. La pendiente de la línea es `b` y `a` es la intersección con el eje Y, que se refiere al valor de `Y` cuando `X = 0`. 
>
>![calcular la pendiente](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/3-Linear/images/slope.png)
>
> Primero, calcula la pendiente `b`. Infografía por [Jen Looper](https://twitter.com/jenlooper)
>
> En otras palabras, y refiriéndonos a la pregunta original de los datos de calabazas: "predecir el precio de una calabaza por bushel según el mes", `X` se referiría al precio y `Y` se referiría al mes de venta. 
>
>![completar la ecuación](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/3-Linear/images/calculation.png)
>
> Calcula el valor de Y. Si estás pagando alrededor de $4, ¡debe ser abril! Infografía por [Jen Looper](https://twitter.com/jenlooper)
>
> Las matemáticas que calculan la línea deben demostrar la pendiente de la línea, que también depende de la intersección, o dónde se sitúa `Y` cuando `X = 0`.
>
> Puedes observar el método de cálculo para estos valores en el sitio web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). También visita [este calculador de mínimos cuadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver cómo los valores de los números afectan la línea.

## Correlación

Otro término que debes entender es el **Coeficiente de Correlación** entre las variables X e Y dadas. Usando un gráfico de dispersión, puedes visualizar rápidamente este coeficiente. Un gráfico con puntos de datos dispersos en una línea ordenada tiene alta correlación, pero un gráfico con puntos de datos dispersos por todas partes entre X e Y tiene baja correlación.

Un buen modelo de regresión lineal será aquel que tenga un Coeficiente de Correlación alto (más cercano a 1 que a 0) utilizando el método de Regresión de Mínimos Cuadrados con una línea de regresión.

✅ Ejecuta el notebook que acompaña esta lección y observa el gráfico de dispersión de Mes a Precio. Según tu interpretación visual del gráfico de dispersión, ¿parece que los datos que asocian Mes con Precio para las ventas de calabazas tienen alta o baja correlación? ¿Cambia eso si usas una medida más detallada en lugar de `Mes`, por ejemplo, *día del año* (es decir, el número de días desde el inicio del año)?

En el código a continuación, asumiremos que hemos limpiado los datos y obtenido un marco de datos llamado `new_pumpkins`, similar al siguiente:

ID | Mes | DíaDelAño | Variedad | Ciudad | Paquete | Precio Bajo | Precio Alto | Precio
---|-----|-----------|----------|--------|---------|-------------|-------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> El código para limpiar los datos está disponible en [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Hemos realizado los mismos pasos de limpieza que en la lección anterior y hemos calculado la columna `DíaDelAño` utilizando la siguiente expresión: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Ahora que tienes una comprensión de las matemáticas detrás de la regresión lineal, vamos a crear un modelo de Regresión para ver si podemos predecir qué paquete de calabazas tendrá los mejores precios. Alguien que compre calabazas para un huerto de calabazas festivo podría querer esta información para optimizar sus compras de paquetes de calabazas para el huerto.

## Buscando correlación

[![ML para principiantes - Buscando correlación: La clave para la regresión lineal](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para principiantes - Buscando correlación: La clave para la regresión lineal")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre la correlación.

De la lección anterior probablemente hayas visto que el precio promedio para diferentes meses se ve así:

<img alt="Precio promedio por mes" src="../2-Data/images/barchart.png" width="50%"/>

Esto sugiere que debería haber alguna correlación, y podemos intentar entrenar un modelo de regresión lineal para predecir la relación entre `Mes` y `Precio`, o entre `DíaDelAño` y `Precio`. Aquí está el gráfico de dispersión que muestra esta última relación:

<img alt="Gráfico de dispersión de Precio vs. Día del Año" src="images/scatter-dayofyear.png" width="50%" /> 

Veamos si hay una correlación usando la función `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Parece que la correlación es bastante pequeña, -0.15 por `Mes` y -0.17 por `DíaDelAño`, pero podría haber otra relación importante. Parece que hay diferentes grupos de precios que corresponden a diferentes variedades de calabazas. Para confirmar esta hipótesis, tracemos cada categoría de calabaza usando un color diferente. Al pasar un parámetro `ax` a la función de trazado de dispersión podemos trazar todos los puntos en el mismo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Gráfico de dispersión de Precio vs. Día del Año" src="images/scatter-dayofyear-color.png" width="50%" /> 

Nuestra investigación sugiere que la variedad tiene más efecto en el precio general que la fecha de venta real. Podemos ver esto con un gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Gráfico de barras de precio vs variedad" src="images/price-by-variety.png" width="50%" /> 

Centrémonos por el momento solo en una variedad de calabaza, el 'tipo pie', y veamos qué efecto tiene la fecha en el precio:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Gráfico de dispersión de Precio vs. Día del Año" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Si ahora calculamos la correlación entre `Precio` y `DíaDelAño` usando la función `corr`, obtendremos algo como `-0.27`, lo que significa que tiene sentido entrenar un modelo predictivo.

> Antes de entrenar un modelo de regresión lineal, es importante asegurarse de que nuestros datos estén limpios. La regresión lineal no funciona bien con valores faltantes, por lo que tiene sentido deshacerse de todas las celdas vacías:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Otra opción sería llenar esos valores vacíos con valores promedio de la columna correspondiente.

## Regresión Lineal Simple

[![ML para principiantes - Regresión Lineal y Polinómica usando Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para principiantes - Regresión Lineal y Polinómica usando Scikit-learn")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre regresión lineal y polinómica.

Para entrenar nuestro modelo de Regresión Lineal, utilizaremos la biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Comenzamos separando los valores de entrada (características) y la salida esperada (etiqueta) en matrices numpy separadas:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Nota que tuvimos que realizar un `reshape` en los datos de entrada para que el paquete de Regresión Lineal los entienda correctamente. La Regresión Lineal espera una matriz 2D como entrada, donde cada fila de la matriz corresponde a un vector de características de entrada. En nuestro caso, dado que solo tenemos una entrada, necesitamos una matriz con forma N×1, donde N es el tamaño del conjunto de datos.

Luego, necesitamos dividir los datos en conjuntos de entrenamiento y prueba, para que podamos validar nuestro modelo después del entrenamiento:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finalmente, entrenar el modelo de Regresión Lineal real toma solo dos líneas de código. Definimos el objeto `LinearRegression` y lo ajustamos a nuestros datos usando el método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

El objeto `LinearRegression` después de ajustarse contiene todos los coeficientes de la regresión, que se pueden acceder usando la propiedad `.coef_`. En nuestro caso, solo hay un coeficiente, que debería estar alrededor de `-0.017`. Esto significa que los precios parecen bajar un poco con el tiempo, pero no demasiado, alrededor de 2 centavos por día. También podemos acceder al punto de intersección de la regresión con el eje Y usando `lin_reg.intercept_`, que estará alrededor de `21` en nuestro caso, indicando el precio al comienzo del año.

Para ver qué tan preciso es nuestro modelo, podemos predecir precios en un conjunto de datos de prueba y luego medir qué tan cerca están nuestras predicciones de los valores esperados. Esto se puede hacer usando la métrica de error cuadrático medio (MSE), que es el promedio de todas las diferencias al cuadrado entre el valor esperado y el predicho.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Nuestro error parece estar en torno a 2 puntos, lo que equivale a ~17%. No es muy bueno. Otro indicador de la calidad del modelo es el **coeficiente de determinación**, que se puede obtener de la siguiente manera:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Si el valor es 0, significa que el modelo no toma en cuenta los datos de entrada y actúa como el *peor predictor lineal*, que simplemente es el valor promedio del resultado. Un valor de 1 significa que podemos predecir perfectamente todos los resultados esperados. En nuestro caso, el coeficiente está alrededor de 0.06, lo cual es bastante bajo.

También podemos graficar los datos de prueba junto con la línea de regresión para ver mejor cómo funciona la regresión en nuestro caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Regresión lineal" src="images/linear-results.png" width="50%" />

## Regresión Polinómica

Otro tipo de Regresión Lineal es la Regresión Polinómica. Aunque a veces existe una relación lineal entre las variables - cuanto mayor es el volumen de la calabaza, mayor es el precio - en otras ocasiones estas relaciones no pueden representarse como un plano o una línea recta.

✅ Aquí hay [algunos ejemplos](https://online.stat.psu.edu/stat501/lesson/9/9.8) de datos que podrían usar Regresión Polinómica.

Observa nuevamente la relación entre Fecha y Precio. ¿Este diagrama de dispersión parece que debería analizarse necesariamente con una línea recta? ¿No pueden fluctuar los precios? En este caso, puedes intentar con regresión polinómica.

✅ Los polinomios son expresiones matemáticas que pueden consistir en una o más variables y coeficientes.

La regresión polinómica crea una línea curva para ajustar mejor los datos no lineales. En nuestro caso, si incluimos una variable `DayOfYear` al cuadrado en los datos de entrada, deberíamos poder ajustar nuestros datos con una curva parabólica, que tendrá un mínimo en cierto punto del año.

Scikit-learn incluye una útil [API de pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para combinar diferentes pasos de procesamiento de datos. Un **pipeline** es una cadena de **estimadores**. En nuestro caso, crearemos un pipeline que primero agrega características polinómicas a nuestro modelo y luego entrena la regresión:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que incluiremos todos los polinomios de segundo grado de los datos de entrada. En nuestro caso, esto simplemente significará `DayOfYear`<sup>2</sup>, pero dado dos variables de entrada X e Y, esto agregará X<sup>2</sup>, XY y Y<sup>2</sup>. También podemos usar polinomios de mayor grado si lo deseamos.

Los pipelines pueden usarse de la misma manera que el objeto original `LinearRegression`, es decir, podemos usar `fit` en el pipeline y luego usar `predict` para obtener los resultados de predicción. Aquí está el gráfico que muestra los datos de prueba y la curva de aproximación:

<img alt="Regresión polinómica" src="images/poly-results.png" width="50%" />

Usando Regresión Polinómica, podemos obtener un MSE ligeramente más bajo y un coeficiente de determinación más alto, pero no significativamente. ¡Necesitamos tomar en cuenta otras características!

> Puedes observar que los precios mínimos de las calabazas se registran en algún momento cerca de Halloween. ¿Cómo puedes explicar esto?

🎃 ¡Felicidades! Acabas de crear un modelo que puede ayudar a predecir el precio de las calabazas para pastel. Probablemente podrías repetir el mismo procedimiento para todos los tipos de calabazas, pero eso sería tedioso. ¡Ahora aprendamos cómo tomar en cuenta la variedad de calabazas en nuestro modelo!

## Características Categóricas

En un mundo ideal, queremos poder predecir precios para diferentes variedades de calabazas usando el mismo modelo. Sin embargo, la columna `Variety` es algo diferente de columnas como `Month`, porque contiene valores no numéricos. Estas columnas se llaman **categóricas**.

[![ML para principiantes - Predicciones con características categóricas usando Regresión Lineal](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML para principiantes - Predicciones con características categóricas usando Regresión Lineal")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre el uso de características categóricas.

Aquí puedes ver cómo el precio promedio depende de la variedad:

<img alt="Precio promedio por variedad" src="images/price-by-variety.png" width="50%" />

Para tomar en cuenta la variedad, primero necesitamos convertirla a forma numérica, o **codificarla**. Hay varias maneras de hacerlo:

* La **codificación numérica simple** construirá una tabla de diferentes variedades y luego reemplazará el nombre de la variedad por un índice en esa tabla. Esta no es la mejor idea para la regresión lineal, porque la regresión lineal toma el valor numérico real del índice y lo agrega al resultado, multiplicándolo por algún coeficiente. En nuestro caso, la relación entre el número de índice y el precio claramente no es lineal, incluso si nos aseguramos de que los índices estén ordenados de alguna manera específica.
* La **codificación one-hot** reemplazará la columna `Variety` por 4 columnas diferentes, una para cada variedad. Cada columna contendrá `1` si la fila correspondiente es de una variedad dada, y `0` en caso contrario. Esto significa que habrá cuatro coeficientes en la regresión lineal, uno para cada variedad de calabaza, responsables del "precio inicial" (o más bien "precio adicional") para esa variedad en particular.

El siguiente código muestra cómo podemos codificar una variedad usando one-hot:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Para entrenar la regresión lineal usando la variedad codificada como one-hot en los datos de entrada, solo necesitamos inicializar correctamente los datos `X` y `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

El resto del código es el mismo que usamos anteriormente para entrenar la Regresión Lineal. Si lo pruebas, verás que el error cuadrático medio es aproximadamente el mismo, pero obtenemos un coeficiente de determinación mucho más alto (~77%). Para obtener predicciones aún más precisas, podemos tomar en cuenta más características categóricas, así como características numéricas, como `Month` o `DayOfYear`. Para obtener un gran conjunto de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aquí también tomamos en cuenta `City` y el tipo de `Package`, lo que nos da un MSE de 2.84 (10%) y un coeficiente de determinación de 0.94.

## Juntándolo todo

Para crear el mejor modelo, podemos usar datos combinados (categóricos codificados como one-hot + numéricos) del ejemplo anterior junto con Regresión Polinómica. Aquí está el código completo para tu conveniencia:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Esto debería darnos el mejor coeficiente de determinación de casi 97% y un MSE=2.23 (~8% de error de predicción).

| Modelo | MSE | Determinación |
|--------|-----|---------------|
| `DayOfYear` Lineal | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polinómico | 2.73 (17.0%) | 0.08 |
| `Variety` Lineal | 5.24 (19.7%) | 0.77 |
| Todas las características Lineal | 2.84 (10.5%) | 0.94 |
| Todas las características Polinómico | 2.23 (8.25%) | 0.97 |

🏆 ¡Bien hecho! Creaste cuatro modelos de Regresión en una sola lección y mejoraste la calidad del modelo al 97%. En la sección final sobre Regresión, aprenderás sobre Regresión Logística para determinar categorías.

---
## 🚀Desafío

Prueba varias variables diferentes en este notebook para ver cómo la correlación corresponde a la precisión del modelo.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

En esta lección aprendimos sobre Regresión Lineal. Hay otros tipos importantes de Regresión. Lee sobre las técnicas Stepwise, Ridge, Lasso y Elasticnet. Un buen curso para estudiar y aprender más es el [curso de Stanford sobre Aprendizaje Estadístico](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Tarea

[Construye un Modelo](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-04T22:12:57+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "es"
}
-->
# Regresión logística para predecir categorías

![Infografía de regresión logística vs. regresión lineal](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Introducción

En esta última lección sobre Regresión, una de las técnicas básicas _clásicas_ de ML, exploraremos la Regresión Logística. Utilizarías esta técnica para descubrir patrones y predecir categorías binarias. ¿Es este dulce de chocolate o no? ¿Es esta enfermedad contagiosa o no? ¿Elegirá este cliente este producto o no?

En esta lección, aprenderás:

- Una nueva biblioteca para la visualización de datos
- Técnicas para la regresión logística

✅ Profundiza tu comprensión sobre cómo trabajar con este tipo de regresión en este [módulo de aprendizaje](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Prerrequisitos

Después de trabajar con los datos de calabazas, ya estamos lo suficientemente familiarizados con ellos como para darnos cuenta de que hay una categoría binaria con la que podemos trabajar: `Color`.

Construyamos un modelo de regresión logística para predecir, dado algunas variables, _de qué color es probable que sea una calabaza_ (naranja 🎃 o blanca 👻).

> ¿Por qué estamos hablando de clasificación binaria en una lección sobre regresión? Solo por conveniencia lingüística, ya que la regresión logística es [realmente un método de clasificación](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), aunque basado en un enfoque lineal. Aprende sobre otras formas de clasificar datos en el próximo grupo de lecciones.

## Define la pregunta

Para nuestros propósitos, expresaremos esto como un binario: 'Blanca' o 'No Blanca'. También hay una categoría 'rayada' en nuestro conjunto de datos, pero hay pocos casos de ella, por lo que no la usaremos. De todos modos, desaparece una vez que eliminamos los valores nulos del conjunto de datos.

> 🎃 Dato curioso: a veces llamamos a las calabazas blancas 'calabazas fantasma'. No son muy fáciles de tallar, por lo que no son tan populares como las naranjas, ¡pero tienen un aspecto genial! Así que también podríamos reformular nuestra pregunta como: 'Fantasma' o 'No Fantasma'. 👻

## Sobre la regresión logística

La regresión logística difiere de la regresión lineal, que aprendiste anteriormente, en algunos aspectos importantes.

[![ML para principiantes - Comprendiendo la Regresión Logística para la Clasificación en Machine Learning](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML para principiantes - Comprendiendo la Regresión Logística para la Clasificación en Machine Learning")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre la regresión logística.

### Clasificación binaria

La regresión logística no ofrece las mismas características que la regresión lineal. La primera ofrece una predicción sobre una categoría binaria ("blanca o no blanca"), mientras que la segunda es capaz de predecir valores continuos, por ejemplo, dado el origen de una calabaza y el momento de la cosecha, _cuánto subirá su precio_.

![Modelo de clasificación de calabazas](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Otras clasificaciones

Existen otros tipos de regresión logística, incluyendo multinomial y ordinal:

- **Multinomial**, que implica tener más de una categoría - "Naranja, Blanca y Rayada".
- **Ordinal**, que implica categorías ordenadas, útil si quisiéramos ordenar nuestros resultados lógicamente, como nuestras calabazas que están ordenadas por un número finito de tamaños (mini, pequeña, mediana, grande, XL, XXL).

![Regresión multinomial vs ordinal](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Las variables NO tienen que correlacionarse

¿Recuerdas cómo la regresión lineal funcionaba mejor con variables más correlacionadas? La regresión logística es lo opuesto: las variables no tienen que estar alineadas. Esto funciona para estos datos, que tienen correlaciones algo débiles.

### Necesitas muchos datos limpios

La regresión logística dará resultados más precisos si utilizas más datos; nuestro pequeño conjunto de datos no es óptimo para esta tarea, así que tenlo en cuenta.

[![ML para principiantes - Análisis y Preparación de Datos para la Regresión Logística](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML para principiantes - Análisis y Preparación de Datos para la Regresión Logística")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre la preparación de datos para la regresión lineal.

✅ Piensa en los tipos de datos que se prestarían bien a la regresión logística.

## Ejercicio - organiza los datos

Primero, limpia un poco los datos, eliminando valores nulos y seleccionando solo algunas de las columnas:

1. Agrega el siguiente código:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Siempre puedes echar un vistazo a tu nuevo dataframe:

    ```python
    pumpkins.info
    ```

### Visualización - gráfico categórico

Hasta ahora has cargado el [notebook inicial](../../../../2-Regression/4-Logistic/notebook.ipynb) con datos de calabazas una vez más y lo has limpiado para preservar un conjunto de datos que contiene algunas variables, incluyendo `Color`. Visualicemos el dataframe en el notebook usando una biblioteca diferente: [Seaborn](https://seaborn.pydata.org/index.html), que está construida sobre Matplotlib, que usamos anteriormente.

Seaborn ofrece formas interesantes de visualizar tus datos. Por ejemplo, puedes comparar distribuciones de los datos para cada `Variety` y `Color` en un gráfico categórico.

1. Crea dicho gráfico usando la función `catplot`, con nuestros datos de calabazas `pumpkins`, y especificando un mapeo de colores para cada categoría de calabaza (naranja o blanca):

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![Una cuadrícula de datos visualizados](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Al observar los datos, puedes ver cómo los datos de Color se relacionan con Variety.

    ✅ Dado este gráfico categórico, ¿qué exploraciones interesantes puedes imaginar?

### Preprocesamiento de datos: codificación de características y etiquetas

Nuestro conjunto de datos de calabazas contiene valores de cadena para todas sus columnas. Trabajar con datos categóricos es intuitivo para los humanos, pero no para las máquinas. Los algoritmos de aprendizaje automático funcionan bien con números. Por eso, la codificación es un paso muy importante en la fase de preprocesamiento de datos, ya que nos permite convertir datos categóricos en datos numéricos, sin perder información. Una buena codificación conduce a la construcción de un buen modelo.

Para la codificación de características, hay dos tipos principales de codificadores:

1. Codificador ordinal: es adecuado para variables ordinales, que son variables categóricas donde sus datos siguen un orden lógico, como la columna `Item Size` en nuestro conjunto de datos. Crea un mapeo de modo que cada categoría esté representada por un número, que es el orden de la categoría en la columna.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Codificador categórico: es adecuado para variables nominales, que son variables categóricas donde sus datos no siguen un orden lógico, como todas las características diferentes de `Item Size` en nuestro conjunto de datos. Es una codificación one-hot, lo que significa que cada categoría está representada por una columna binaria: la variable codificada es igual a 1 si la calabaza pertenece a esa Variety y 0 en caso contrario.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Luego, `ColumnTransformer` se utiliza para combinar múltiples codificadores en un solo paso y aplicarlos a las columnas apropiadas.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

Por otro lado, para codificar la etiqueta, usamos la clase `LabelEncoder` de scikit-learn, que es una clase de utilidad para ayudar a normalizar etiquetas de modo que contengan solo valores entre 0 y n_classes-1 (aquí, 0 y 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

Una vez que hemos codificado las características y la etiqueta, podemos fusionarlas en un nuevo dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ ¿Cuáles son las ventajas de usar un codificador ordinal para la columna `Item Size`?

### Analiza las relaciones entre variables

Ahora que hemos preprocesado nuestros datos, podemos analizar las relaciones entre las características y la etiqueta para hacernos una idea de qué tan bien el modelo podrá predecir la etiqueta dadas las características. La mejor manera de realizar este tipo de análisis es graficando los datos. Usaremos nuevamente la función `catplot` de Seaborn para visualizar las relaciones entre `Item Size`, `Variety` y `Color` en un gráfico categórico. Para graficar mejor los datos, usaremos la columna codificada `Item Size` y la columna sin codificar `Variety`.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```

![Un gráfico categórico de datos visualizados](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Usa un gráfico de enjambre

Dado que Color es una categoría binaria (Blanca o No), necesita '[un enfoque especializado](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) para la visualización'. Hay otras formas de visualizar la relación de esta categoría con otras variables.

Puedes visualizar variables lado a lado con gráficos de Seaborn.

1. Prueba un gráfico de 'enjambre' para mostrar la distribución de valores:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Un enjambre de datos visualizados](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/swarm_2.png)

**Cuidado**: el código anterior podría generar una advertencia, ya que Seaborn falla al representar tal cantidad de puntos de datos en un gráfico de enjambre. Una posible solución es disminuir el tamaño del marcador, utilizando el parámetro 'size'. Sin embargo, ten en cuenta que esto afecta la legibilidad del gráfico.

> **🧮 Muéstrame las matemáticas**
>
> La regresión logística se basa en el concepto de 'máxima verosimilitud' utilizando [funciones sigmoides](https://wikipedia.org/wiki/Sigmoid_function). Una 'Función Sigmoide' en un gráfico tiene forma de 'S'. Toma un valor y lo mapea a un rango entre 0 y 1. Su curva también se llama 'curva logística'. Su fórmula es la siguiente:
>
> ![función logística](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/sigmoid.png)
>
> donde el punto medio de la sigmoide se encuentra en el punto 0 de x, L es el valor máximo de la curva, y k es la pendiente de la curva. Si el resultado de la función es mayor a 0.5, la etiqueta en cuestión se clasificará como '1' de la elección binaria. Si no, se clasificará como '0'.

## Construye tu modelo

Construir un modelo para encontrar estas clasificaciones binarias es sorprendentemente sencillo en Scikit-learn.

[![ML para principiantes - Regresión Logística para la clasificación de datos](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML para principiantes - Regresión Logística para la clasificación de datos")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre cómo construir un modelo de regresión lineal.

1. Selecciona las variables que deseas usar en tu modelo de clasificación y divide los conjuntos de entrenamiento y prueba llamando a `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Ahora puedes entrenar tu modelo llamando a `fit()` con tus datos de entrenamiento y mostrar su resultado:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    Observa el puntaje de tu modelo. No está mal, considerando que solo tienes alrededor de 1000 filas de datos:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## Mejor comprensión mediante una matriz de confusión

Aunque puedes obtener un informe de puntaje [términos](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) imprimiendo los elementos anteriores, podrías entender mejor tu modelo utilizando una [matriz de confusión](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) para ayudarnos a comprender cómo está funcionando el modelo.

> 🎓 Una '[matriz de confusión](https://wikipedia.org/wiki/Confusion_matrix)' (o 'matriz de error') es una tabla que expresa los verdaderos vs. falsos positivos y negativos de tu modelo, evaluando así la precisión de las predicciones.

1. Para usar una matriz de confusión, llama a `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Observa la matriz de confusión de tu modelo:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

En Scikit-learn, las filas (eje 0) son etiquetas reales y las columnas (eje 1) son etiquetas predichas.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

¿Qué está pasando aquí? Supongamos que nuestro modelo debe clasificar calabazas entre dos categorías binarias, categoría 'blanca' y categoría 'no blanca'.

- Si tu modelo predice una calabaza como no blanca y en realidad pertenece a la categoría 'no blanca', lo llamamos un verdadero negativo, mostrado por el número en la esquina superior izquierda.
- Si tu modelo predice una calabaza como blanca y en realidad pertenece a la categoría 'no blanca', lo llamamos un falso negativo, mostrado por el número en la esquina inferior izquierda.
- Si tu modelo predice una calabaza como no blanca y en realidad pertenece a la categoría 'blanca', lo llamamos un falso positivo, mostrado por el número en la esquina superior derecha.
- Si tu modelo predice una calabaza como blanca y en realidad pertenece a la categoría 'blanca', lo llamamos un verdadero positivo, mostrado por el número en la esquina inferior derecha.

Como habrás adivinado, es preferible tener un mayor número de verdaderos positivos y verdaderos negativos y un menor número de falsos positivos y falsos negativos, lo que implica que el modelo funciona mejor.
¿Cómo se relaciona la matriz de confusión con la precisión y el recall? Recuerda, el informe de clasificación mostrado anteriormente indicó una precisión (0.85) y un recall (0.67).

Precisión = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Recall = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ P: Según la matriz de confusión, ¿cómo le fue al modelo? R: No está mal; hay un buen número de verdaderos negativos, pero también algunos falsos negativos.

Volvamos a revisar los términos que vimos antes con la ayuda del mapeo de TP/TN y FP/FN en la matriz de confusión:

🎓 Precisión: TP/(TP + FP) La fracción de instancias relevantes entre las instancias recuperadas (por ejemplo, qué etiquetas fueron bien etiquetadas).

🎓 Recall: TP/(TP + FN) La fracción de instancias relevantes que fueron recuperadas, ya sea bien etiquetadas o no.

🎓 f1-score: (2 * precisión * recall)/(precisión + recall) Un promedio ponderado de la precisión y el recall, donde el mejor valor es 1 y el peor es 0.

🎓 Soporte: El número de ocurrencias de cada etiqueta recuperada.

🎓 Exactitud: (TP + TN)/(TP + TN + FP + FN) El porcentaje de etiquetas predichas correctamente para una muestra.

🎓 Promedio Macro: El cálculo de las métricas medias no ponderadas para cada etiqueta, sin tener en cuenta el desequilibrio de etiquetas.

🎓 Promedio Ponderado: El cálculo de las métricas medias para cada etiqueta, teniendo en cuenta el desequilibrio de etiquetas al ponderarlas según su soporte (el número de instancias verdaderas para cada etiqueta).

✅ ¿Puedes pensar en qué métrica deberías enfocarte si quieres que tu modelo reduzca el número de falsos negativos?

## Visualizar la curva ROC de este modelo

[![ML para principiantes - Analizando el rendimiento de la regresión logística con curvas ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML para principiantes - Analizando el rendimiento de la regresión logística con curvas ROC")

> 🎥 Haz clic en la imagen de arriba para un breve video sobre las curvas ROC.

Hagamos una visualización más para observar la llamada curva 'ROC':

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Usando Matplotlib, grafica la [Curva Característica Operativa del Receptor](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) o ROC del modelo. Las curvas ROC se usan a menudo para obtener una vista del rendimiento de un clasificador en términos de sus verdaderos positivos frente a los falsos positivos. "Las curvas ROC típicamente muestran la tasa de verdaderos positivos en el eje Y y la tasa de falsos positivos en el eje X." Por lo tanto, la inclinación de la curva y el espacio entre la línea del punto medio y la curva son importantes: quieres una curva que suba rápidamente y se aleje de la línea. En nuestro caso, hay falsos positivos al principio, y luego la línea sube y se aleja correctamente:

![ROC](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/images/ROC_2.png)

Finalmente, usa la API [`roc_auc_score` de Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) para calcular el 'Área Bajo la Curva' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
El resultado es `0.9749908725812341`. Dado que el AUC varía de 0 a 1, quieres un puntaje alto, ya que un modelo que sea 100% correcto en sus predicciones tendrá un AUC de 1; en este caso, el modelo es _bastante bueno_.

En futuras lecciones sobre clasificaciones, aprenderás cómo iterar para mejorar los puntajes de tu modelo. Pero por ahora, ¡felicitaciones! ¡Has completado estas lecciones sobre regresión!

---
## 🚀Desafío

¡Hay mucho más que explorar sobre la regresión logística! Pero la mejor manera de aprender es experimentando. Encuentra un conjunto de datos que se preste a este tipo de análisis y construye un modelo con él. ¿Qué aprendes? Consejo: prueba [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) para encontrar conjuntos de datos interesantes.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

Lee las primeras páginas de [este artículo de Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) sobre algunos usos prácticos de la regresión logística. Piensa en tareas que se adapten mejor a uno u otro tipo de tareas de regresión que hemos estudiado hasta ahora. ¿Qué funcionaría mejor?

## Tarea

[Reintentando esta regresión](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---
