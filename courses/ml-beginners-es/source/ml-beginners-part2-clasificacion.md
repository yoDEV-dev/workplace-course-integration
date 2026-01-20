# ML para Principiantes 🤖
## Parte 2: Clasificación
**Aplicaciones Web, Clasificación y Clustering**

---


# Aplicaciones Web

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T23:43:22+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "es"
}
-->
# Crea una aplicación web para usar tu modelo de ML

En esta sección del curso, se te presentará un tema práctico de aprendizaje automático: cómo guardar tu modelo de Scikit-learn como un archivo que pueda ser utilizado para hacer predicciones dentro de una aplicación web. Una vez que el modelo esté guardado, aprenderás cómo usarlo en una aplicación web construida con Flask. Primero, crearás un modelo utilizando algunos datos relacionados con avistamientos de OVNIs. Luego, construirás una aplicación web que te permitirá ingresar un número de segundos junto con un valor de latitud y longitud para predecir qué país reportó haber visto un OVNI.

![UFO Parking](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/ufo.9e787f5161da9d4d.webp)

Foto por <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> en <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lecciones

1. [Crea una aplicación web](1-Web-App/README.md)

## Créditos

"Crea una aplicación web" fue escrito con ♥️ por [Jen Looper](https://twitter.com/jenlooper).

♥️ Los cuestionarios fueron escritos por Rohan Raj.

El conjunto de datos proviene de [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

La arquitectura de la aplicación web fue sugerida en parte por [este artículo](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) y [este repositorio](https://github.com/abhinavsagar/machine-learning-deployment) de Abhinav Sagar.

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-04T22:22:36+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "es"
}
-->
# Construir una aplicación web para usar un modelo de ML

En esta lección, entrenarás un modelo de ML con un conjunto de datos que es de otro mundo: _avistamientos de OVNIs durante el último siglo_, obtenidos de la base de datos de NUFORC.

Aprenderás:

- Cómo 'pickle' un modelo entrenado
- Cómo usar ese modelo en una aplicación Flask

Continuaremos utilizando notebooks para limpiar datos y entrenar nuestro modelo, pero puedes llevar el proceso un paso más allá explorando cómo usar un modelo 'en el mundo real', por así decirlo: en una aplicación web.

Para hacer esto, necesitas construir una aplicación web usando Flask.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Construyendo una aplicación

Existen varias formas de construir aplicaciones web para consumir modelos de aprendizaje automático. La arquitectura de tu aplicación web puede influir en la forma en que se entrena tu modelo. Imagina que trabajas en una empresa donde el grupo de ciencia de datos ha entrenado un modelo que quieren que utilices en una aplicación.

### Consideraciones

Hay muchas preguntas que debes hacerte:

- **¿Es una aplicación web o una aplicación móvil?** Si estás construyendo una aplicación móvil o necesitas usar el modelo en un contexto de IoT, podrías usar [TensorFlow Lite](https://www.tensorflow.org/lite/) y usar el modelo en una aplicación Android o iOS.
- **¿Dónde residirá el modelo?** ¿En la nube o localmente?
- **Soporte sin conexión.** ¿La aplicación necesita funcionar sin conexión?
- **¿Qué tecnología se utilizó para entrenar el modelo?** La tecnología elegida puede influir en las herramientas que necesitas usar.
    - **Usando TensorFlow.** Si estás entrenando un modelo usando TensorFlow, por ejemplo, ese ecosistema proporciona la capacidad de convertir un modelo de TensorFlow para usarlo en una aplicación web mediante [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Si estás construyendo un modelo usando una biblioteca como [PyTorch](https://pytorch.org/), tienes la opción de exportarlo en formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para usarlo en aplicaciones web JavaScript que pueden usar [Onnx Runtime](https://www.onnxruntime.ai/). Esta opción será explorada en una lección futura para un modelo entrenado con Scikit-learn.
    - **Usando Lobe.ai o Azure Custom Vision.** Si estás utilizando un sistema SaaS (Software como Servicio) de ML como [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para entrenar un modelo, este tipo de software proporciona formas de exportar el modelo para muchas plataformas, incluyendo la construcción de una API personalizada para ser consultada en la nube por tu aplicación en línea.

También tienes la oportunidad de construir una aplicación web completa en Flask que pueda entrenar el modelo directamente en un navegador web. Esto también se puede hacer usando TensorFlow.js en un contexto de JavaScript.

Para nuestros propósitos, dado que hemos estado trabajando con notebooks basados en Python, exploremos los pasos que necesitas seguir para exportar un modelo entrenado desde dicho notebook a un formato legible por una aplicación web construida en Python.

## Herramienta

Para esta tarea, necesitas dos herramientas: Flask y Pickle, ambas ejecutándose en Python.

✅ ¿Qué es [Flask](https://palletsprojects.com/p/flask/)? Definido como un 'micro-framework' por sus creadores, Flask proporciona las características básicas de los frameworks web usando Python y un motor de plantillas para construir páginas web. Echa un vistazo a [este módulo de aprendizaje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para practicar la construcción con Flask.

✅ ¿Qué es [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 es un módulo de Python que serializa y deserializa una estructura de objetos de Python. Cuando 'pickleas' un modelo, serializas o aplanas su estructura para usarlo en la web. Ten cuidado: pickle no es intrínsecamente seguro, así que ten precaución si se te solicita 'despicklear' un archivo. Un archivo pickled tiene el sufijo `.pkl`.

## Ejercicio - limpia tus datos

En esta lección usarás datos de 80,000 avistamientos de OVNIs, recopilados por [NUFORC](https://nuforc.org) (El Centro Nacional de Reportes de OVNIs). Estos datos tienen descripciones interesantes de avistamientos de OVNIs, por ejemplo:

- **Descripción larga de ejemplo.** "Un hombre emerge de un rayo de luz que ilumina un campo de hierba por la noche y corre hacia el estacionamiento de Texas Instruments".
- **Descripción corta de ejemplo.** "las luces nos persiguieron".

La hoja de cálculo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) incluye columnas sobre la `ciudad`, `estado` y `país` donde ocurrió el avistamiento, la `forma` del objeto y su `latitud` y `longitud`.

En el [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) en blanco incluido en esta lección:

1. Importa `pandas`, `matplotlib` y `numpy` como lo hiciste en lecciones anteriores e importa la hoja de cálculo de ufos. Puedes echar un vistazo a un conjunto de datos de muestra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convierte los datos de ufos a un pequeño dataframe con títulos nuevos. Revisa los valores únicos en el campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ahora, puedes reducir la cantidad de datos con los que necesitamos trabajar eliminando cualquier valor nulo e importando solo avistamientos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa la biblioteca `LabelEncoder` de Scikit-learn para convertir los valores de texto de los países a un número:

    ✅ LabelEncoder codifica datos alfabéticamente

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Tus datos deberían verse así:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Ejercicio - construye tu modelo

Ahora puedes prepararte para entrenar un modelo dividiendo los datos en grupos de entrenamiento y prueba.

1. Selecciona las tres características que deseas entrenar como tu vector X, y el vector y será el `Country`. Quieres poder ingresar `Seconds`, `Latitude` y `Longitude` y obtener un id de país como resultado.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Entrena tu modelo usando regresión logística:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

La precisión no está mal **(alrededor del 95%)**, lo cual no es sorprendente, ya que `Country` y `Latitude/Longitude` están correlacionados.

El modelo que creaste no es muy revolucionario, ya que deberías poder inferir un `Country` a partir de su `Latitude` y `Longitude`, pero es un buen ejercicio para intentar entrenar desde datos en bruto que limpiaste, exportaste y luego usar este modelo en una aplicación web.

## Ejercicio - 'pickle' tu modelo

¡Ahora es momento de _picklear_ tu modelo! Puedes hacerlo en unas pocas líneas de código. Una vez que esté _pickled_, carga tu modelo pickled y pruébalo contra un arreglo de datos de muestra que contenga valores para segundos, latitud y longitud.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

El modelo devuelve **'3'**, que es el código de país para el Reino Unido. ¡Increíble! 👽

## Ejercicio - construye una aplicación Flask

Ahora puedes construir una aplicación Flask para llamar a tu modelo y devolver resultados similares, pero de una manera más visualmente atractiva.

1. Comienza creando una carpeta llamada **web-app** junto al archivo _notebook.ipynb_ donde reside tu archivo _ufo-model.pkl_.

1. En esa carpeta crea tres carpetas más: **static**, con una carpeta **css** dentro de ella, y **templates**. Ahora deberías tener los siguientes archivos y directorios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta la carpeta de solución para ver la aplicación terminada

1. El primer archivo que debes crear en la carpeta _web-app_ es el archivo **requirements.txt**. Como _package.json_ en una aplicación JavaScript, este archivo lista las dependencias requeridas por la aplicación. En **requirements.txt** agrega las líneas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ahora, ejecuta este archivo navegando a _web-app_:

    ```bash
    cd web-app
    ```

1. En tu terminal escribe `pip install`, para instalar las bibliotecas listadas en _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ahora, estás listo para crear tres archivos más para terminar la aplicación:

    1. Crea **app.py** en la raíz.
    2. Crea **index.html** en el directorio _templates_.
    3. Crea **styles.css** en el directorio _static/css_.

1. Construye el archivo _styles.css_ con algunos estilos:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. A continuación, construye el archivo _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Observa la plantilla en este archivo. Nota la sintaxis de 'bigotes' alrededor de las variables que serán proporcionadas por la aplicación, como el texto de predicción: `{{}}`. También hay un formulario que envía una predicción a la ruta `/predict`.

    Finalmente, estás listo para construir el archivo Python que impulsa el consumo del modelo y la visualización de las predicciones:

1. En `app.py` agrega:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Consejo: cuando agregas [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) mientras ejecutas la aplicación web usando Flask, cualquier cambio que hagas en tu aplicación se reflejará inmediatamente sin necesidad de reiniciar el servidor. ¡Cuidado! No habilites este modo en una aplicación de producción.

Si ejecutas `python app.py` o `python3 app.py`, tu servidor web se iniciará localmente y podrás completar un formulario corto para obtener una respuesta a tu pregunta sobre dónde se han avistado OVNIs.

Antes de hacer eso, echa un vistazo a las partes de `app.py`:

1. Primero, se cargan las dependencias y se inicia la aplicación.
1. Luego, se importa el modelo.
1. Después, se renderiza index.html en la ruta principal.

En la ruta `/predict`, suceden varias cosas cuando se envía el formulario:

1. Las variables del formulario se recopilan y se convierten en un arreglo numpy. Luego se envían al modelo y se devuelve una predicción.
2. Los países que queremos mostrar se vuelven a renderizar como texto legible a partir de su código de país predicho, y ese valor se envía de vuelta a index.html para ser renderizado en la plantilla.

Usar un modelo de esta manera, con Flask y un modelo pickled, es relativamente sencillo. Lo más difícil es entender qué forma deben tener los datos que se deben enviar al modelo para obtener una predicción. Todo depende de cómo se entrenó el modelo. Este tiene tres puntos de datos que deben ingresarse para obtener una predicción.

En un entorno profesional, puedes ver cómo es necesaria una buena comunicación entre las personas que entrenan el modelo y las que lo consumen en una aplicación web o móvil. En nuestro caso, ¡es solo una persona, tú!

---

## 🚀 Desafío

En lugar de trabajar en un notebook e importar el modelo a la aplicación Flask, podrías entrenar el modelo directamente dentro de la aplicación Flask. Intenta convertir tu código Python en el notebook, tal vez después de que tus datos estén limpios, para entrenar el modelo desde dentro de la aplicación en una ruta llamada `train`. ¿Cuáles son las ventajas y desventajas de seguir este método?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y autoestudio

Existen muchas formas de construir una aplicación web para consumir modelos de ML. Haz una lista de las formas en que podrías usar JavaScript o Python para construir una aplicación web que aproveche el aprendizaje automático. Considera la arquitectura: ¿debería el modelo permanecer en la aplicación o vivir en la nube? Si es lo último, ¿cómo lo accederías? Dibuja un modelo arquitectónico para una solución web de ML aplicada.

## Tarea

[Prueba un modelo diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---


# Clasificación

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "74e809ffd1e613a1058bbc3e9600859e",
  "translation_date": "2025-09-03T23:49:12+00:00",
  "source_file": "4-Classification/README.md",
  "language_code": "es"
}
-->
# Comenzando con la clasificación

## Tema regional: Deliciosas cocinas asiáticas e indias 🍜

En Asia e India, las tradiciones culinarias son extremadamente diversas y ¡muy deliciosas! Vamos a analizar datos sobre cocinas regionales para tratar de entender sus ingredientes.

![Vendedor de comida tailandesa](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/thai-food.c47a7a7f9f05c218.webp)
> Foto de <a href="https://unsplash.com/@changlisheng?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Lisheng Chang</a> en <a href="https://unsplash.com/s/photos/asian-food?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Lo que aprenderás

En esta sección, ampliarás tu estudio previo sobre Regresión y aprenderás sobre otros clasificadores que puedes usar para comprender mejor los datos.

> Existen herramientas útiles de bajo código que pueden ayudarte a aprender a trabajar con modelos de clasificación. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-classification-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecciones

1. [Introducción a la clasificación](1-Introduction/README.md)
2. [Más clasificadores](2-Classifiers-1/README.md)
3. [Otros clasificadores](3-Classifiers-2/README.md)
4. [ML aplicado: construir una aplicación web](4-Applied/README.md)

## Créditos

"Comenzando con la clasificación" fue escrito con ♥️ por [Cassie Breviu](https://www.twitter.com/cassiebreviu) y [Jen Looper](https://www.twitter.com/jenlooper)

El conjunto de datos sobre deliciosas cocinas fue obtenido de [Kaggle](https://www.kaggle.com/hoandan/asian-and-indian-cuisines).

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-04T22:24:24+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "es"
}
-->
# Introducción a la clasificación

En estas cuatro lecciones, explorarás un enfoque fundamental del aprendizaje automático clásico: _la clasificación_. Utilizaremos varios algoritmos de clasificación con un conjunto de datos sobre las brillantes cocinas de Asia e India. ¡Espero que tengas hambre!

![solo una pizca!](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/pinch.png)

> Celebra las cocinas panasiáticas en estas lecciones. Imagen de [Jen Looper](https://twitter.com/jenlooper)

La clasificación es una forma de [aprendizaje supervisado](https://wikipedia.org/wiki/Supervised_learning) que tiene mucho en común con las técnicas de regresión. Si el aprendizaje automático se trata de predecir valores o nombres de cosas utilizando conjuntos de datos, entonces la clasificación generalmente se divide en dos grupos: _clasificación binaria_ y _clasificación multiclase_.

[![Introducción a la clasificación](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introducción a la clasificación")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT introduce la clasificación.

Recuerda:

- **Regresión lineal** te ayudó a predecir relaciones entre variables y hacer predicciones precisas sobre dónde caería un nuevo punto de datos en relación con esa línea. Por ejemplo, podrías predecir _qué precio tendría una calabaza en septiembre frente a diciembre_.
- **Regresión logística** te ayudó a descubrir "categorías binarias": a este precio, _¿esta calabaza es naranja o no naranja_?

La clasificación utiliza varios algoritmos para determinar otras formas de asignar una etiqueta o clase a un punto de datos. Trabajemos con estos datos de cocina para ver si, al observar un grupo de ingredientes, podemos determinar su cocina de origen.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introducción

La clasificación es una de las actividades fundamentales del investigador de aprendizaje automático y del científico de datos. Desde la clasificación básica de un valor binario ("¿este correo electrónico es spam o no?"), hasta la clasificación y segmentación compleja de imágenes utilizando visión por computadora, siempre es útil poder ordenar datos en clases y hacer preguntas sobre ellos.

Para expresar el proceso de manera más científica, tu método de clasificación crea un modelo predictivo que te permite mapear la relación entre las variables de entrada y las variables de salida.

![clasificación binaria vs. multiclase](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/binary-multiclass.png)

> Problemas binarios vs. multiclase para que los algoritmos de clasificación los manejen. Infografía de [Jen Looper](https://twitter.com/jenlooper)

Antes de comenzar el proceso de limpiar nuestros datos, visualizarlos y prepararlos para nuestras tareas de aprendizaje automático, aprendamos un poco sobre las diversas formas en que el aprendizaje automático puede ser utilizado para clasificar datos.

Derivado de [estadística](https://wikipedia.org/wiki/Statistical_classification), la clasificación utilizando aprendizaje automático clásico utiliza características como `smoker`, `weight` y `age` para determinar _la probabilidad de desarrollar X enfermedad_. Como técnica de aprendizaje supervisado similar a los ejercicios de regresión que realizaste anteriormente, tus datos están etiquetados y los algoritmos de aprendizaje automático utilizan esas etiquetas para clasificar y predecir clases (o 'características') de un conjunto de datos y asignarlas a un grupo o resultado.

✅ Tómate un momento para imaginar un conjunto de datos sobre cocinas. ¿Qué podría responder un modelo multiclase? ¿Qué podría responder un modelo binario? ¿Qué pasaría si quisieras determinar si una cocina dada probablemente utiliza fenogreco? ¿Qué pasaría si quisieras ver si, dado un regalo de una bolsa de supermercado llena de anís estrellado, alcachofas, coliflor y rábano picante, podrías crear un plato típico indio?

[![Cestas misteriosas locas](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Cestas misteriosas locas")

> 🎥 Haz clic en la imagen de arriba para ver un video. Toda la premisa del programa 'Chopped' es la 'cesta misteriosa', donde los chefs tienen que hacer un plato con una selección aleatoria de ingredientes. ¡Seguramente un modelo de aprendizaje automático habría ayudado!

## Hola 'clasificador'

La pregunta que queremos hacer sobre este conjunto de datos de cocina es en realidad una **pregunta multiclase**, ya que tenemos varias cocinas nacionales potenciales con las que trabajar. Dado un lote de ingredientes, ¿a cuál de estas muchas clases se ajustará el dato?

Scikit-learn ofrece varios algoritmos diferentes para clasificar datos, dependiendo del tipo de problema que quieras resolver. En las próximas dos lecciones, aprenderás sobre varios de estos algoritmos.

## Ejercicio - limpia y equilibra tus datos

La primera tarea, antes de comenzar este proyecto, es limpiar y **equilibrar** tus datos para obtener mejores resultados. Comienza con el archivo en blanco _notebook.ipynb_ en la raíz de esta carpeta.

Lo primero que debes instalar es [imblearn](https://imbalanced-learn.org/stable/). Este es un paquete de Scikit-learn que te permitirá equilibrar mejor los datos (aprenderás más sobre esta tarea en un momento).

1. Para instalar `imblearn`, ejecuta `pip install`, de esta manera:

    ```python
    pip install imblearn
    ```

1. Importa los paquetes necesarios para importar tus datos y visualizarlos, también importa `SMOTE` desde `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Ahora estás listo para importar los datos.

1. La siguiente tarea será importar los datos:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Usar `read_csv()` leerá el contenido del archivo csv _cusines.csv_ y lo colocará en la variable `df`.

1. Verifica la forma de los datos:

    ```python
    df.head()
    ```

   Las primeras cinco filas se ven así:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Obtén información sobre estos datos llamando a `info()`:

    ```python
    df.info()
    ```

    Tu salida se parece a:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Ejercicio - aprendiendo sobre cocinas

Ahora el trabajo comienza a ponerse más interesante. Descubramos la distribución de datos por cocina.

1. Grafica los datos como barras llamando a `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribución de datos de cocina](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/cuisine-dist.png)

    Hay un número finito de cocinas, pero la distribución de datos es desigual. ¡Puedes arreglar eso! Antes de hacerlo, explora un poco más.

1. Descubre cuántos datos hay disponibles por cocina e imprímelos:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    La salida se ve así:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Descubriendo ingredientes

Ahora puedes profundizar en los datos y aprender cuáles son los ingredientes típicos por cocina. Deberías limpiar los datos recurrentes que crean confusión entre cocinas, así que aprendamos sobre este problema.

1. Crea una función `create_ingredient()` en Python para crear un marco de datos de ingredientes. Esta función comenzará eliminando una columna poco útil y ordenará los ingredientes por su cantidad:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Ahora puedes usar esa función para obtener una idea de los diez ingredientes más populares por cocina.

1. Llama a `create_ingredient()` y gráficalo llamando a `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/thai.png)

1. Haz lo mismo para los datos japoneses:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/japanese.png)

1. Ahora para los ingredientes chinos:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/chinese.png)

1. Grafica los ingredientes indios:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/indian.png)

1. Finalmente, grafica los ingredientes coreanos:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/korean.png)

1. Ahora, elimina los ingredientes más comunes que crean confusión entre cocinas distintas, llamando a `drop()`:

   ¡A todos les encanta el arroz, el ajo y el jengibre!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Equilibra el conjunto de datos

Ahora que has limpiado los datos, utiliza [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Técnica de Sobremuestreo de Minorías Sintéticas" - para equilibrarlos.

1. Llama a `fit_resample()`, esta estrategia genera nuevas muestras mediante interpolación.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Al equilibrar tus datos, obtendrás mejores resultados al clasificarlos. Piensa en una clasificación binaria. Si la mayoría de tus datos pertenecen a una clase, un modelo de aprendizaje automático va a predecir esa clase con más frecuencia, simplemente porque hay más datos para ella. El equilibrio de los datos elimina este sesgo.

1. Ahora puedes verificar los números de etiquetas por ingrediente:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Tu salida se parece a:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Los datos están limpios, equilibrados y muy deliciosos.

1. El último paso es guardar tus datos equilibrados, incluyendo etiquetas y características, en un nuevo marco de datos que pueda exportarse a un archivo:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Puedes echar un último vistazo a los datos usando `transformed_df.head()` y `transformed_df.info()`. Guarda una copia de estos datos para usarlos en futuras lecciones:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Este nuevo CSV ahora se encuentra en la carpeta de datos raíz.

---

## 🚀Desafío

Este currículo contiene varios conjuntos de datos interesantes. Explora las carpetas `data` y ve si alguna contiene conjuntos de datos que serían apropiados para clasificación binaria o multiclase. ¿Qué preguntas harías sobre este conjunto de datos?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Explora la API de SMOTE. ¿Para qué casos de uso es más adecuada? ¿Qué problemas resuelve?

## Tarea 

[Explora métodos de clasificación](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-04T22:23:06+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "es"
}
-->
# Clasificadores de cocina 1

En esta lección, usarás el conjunto de datos que guardaste en la última lección, lleno de datos equilibrados y limpios sobre cocinas.

Utilizarás este conjunto de datos con una variedad de clasificadores para _predecir una cocina nacional dada un grupo de ingredientes_. Mientras lo haces, aprenderás más sobre algunas de las formas en que los algoritmos pueden ser aprovechados para tareas de clasificación.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)
# Preparación

Asumiendo que completaste [Lección 1](../1-Introduction/README.md), asegúrate de que exista un archivo _cleaned_cuisines.csv_ en la carpeta raíz `/data` para estas cuatro lecciones.

## Ejercicio - predecir una cocina nacional

1. Trabajando en la carpeta _notebook.ipynb_ de esta lección, importa ese archivo junto con la biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Los datos se ven así:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Ahora, importa varias bibliotecas más:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divide las coordenadas X e y en dos dataframes para entrenamiento. `cuisine` puede ser el dataframe de etiquetas:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Se verá así:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Elimina la columna `Unnamed: 0` y la columna `cuisine`, usando `drop()`. Guarda el resto de los datos como características entrenables:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Tus características se ven así:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

¡Ahora estás listo para entrenar tu modelo!

## Elegir tu clasificador

Ahora que tus datos están limpios y listos para el entrenamiento, debes decidir qué algoritmo usar para la tarea.

Scikit-learn agrupa la clasificación bajo Aprendizaje Supervisado, y en esa categoría encontrarás muchas formas de clasificar. [La variedad](https://scikit-learn.org/stable/supervised_learning.html) puede ser bastante abrumadora a primera vista. Los siguientes métodos incluyen técnicas de clasificación:

- Modelos Lineales
- Máquinas de Vectores de Soporte
- Descenso de Gradiente Estocástico
- Vecinos Más Cercanos
- Procesos Gaussianos
- Árboles de Decisión
- Métodos de Ensamble (clasificador por votación)
- Algoritmos multicategoría y multioutput (clasificación multicategoría y multilabel, clasificación multicategoría-multioutput)

> También puedes usar [redes neuronales para clasificar datos](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), pero eso está fuera del alcance de esta lección.

### ¿Qué clasificador elegir?

Entonces, ¿qué clasificador deberías elegir? A menudo, probar varios y buscar un buen resultado es una forma de evaluar. Scikit-learn ofrece una [comparación lado a lado](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) en un conjunto de datos creado, comparando KNeighbors, SVC de dos maneras, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB y QuadraticDiscriminationAnalysis, mostrando los resultados visualizados:

![comparación de clasificadores](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/comparison.png)
> Gráficos generados en la documentación de Scikit-learn

> AutoML resuelve este problema de manera eficiente al realizar estas comparaciones en la nube, permitiéndote elegir el mejor algoritmo para tus datos. Pruébalo [aquí](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Un enfoque mejor

Una mejor manera que adivinar al azar, sin embargo, es seguir las ideas en esta descargable [hoja de trucos de ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Aquí, descubrimos que, para nuestro problema multicategoría, tenemos algunas opciones:

![hoja de trucos para problemas multicategoría](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Una sección de la Hoja de Trucos de Algoritmos de Microsoft, detallando opciones de clasificación multicategoría

✅ Descarga esta hoja de trucos, imprímela y cuélgala en tu pared.

### Razonamiento

Veamos si podemos razonar sobre diferentes enfoques dados los límites que tenemos:

- **Las redes neuronales son demasiado pesadas**. Dado nuestro conjunto de datos limpio pero mínimo, y el hecho de que estamos ejecutando el entrenamiento localmente a través de notebooks, las redes neuronales son demasiado pesadas para esta tarea.
- **No usamos clasificadores de dos clases**. No usamos un clasificador de dos clases, por lo que descartamos el enfoque uno-contra-todos.
- **Un árbol de decisión o regresión logística podrían funcionar**. Un árbol de decisión podría funcionar, o regresión logística para datos multicategoría.
- **Los árboles de decisión potenciados multicategoría resuelven un problema diferente**. El árbol de decisión potenciado multicategoría es más adecuado para tareas no paramétricas, por ejemplo, tareas diseñadas para construir rankings, por lo que no es útil para nosotros.

### Usando Scikit-learn 

Usaremos Scikit-learn para analizar nuestros datos. Sin embargo, hay muchas formas de usar regresión logística en Scikit-learn. Echa un vistazo a los [parámetros que puedes pasar](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Esencialmente hay dos parámetros importantes - `multi_class` y `solver` - que necesitamos especificar cuando pedimos a Scikit-learn que realice una regresión logística. El valor de `multi_class` aplica un cierto comportamiento. El valor del solver es el algoritmo que se usará. No todos los solvers pueden ser emparejados con todos los valores de `multi_class`.

Según la documentación, en el caso multicategoría, el algoritmo de entrenamiento:

- **Usa el esquema uno-contra-resto (OvR)**, si la opción `multi_class` está configurada como `ovr`.
- **Usa la pérdida de entropía cruzada**, si la opción `multi_class` está configurada como `multinomial`. (Actualmente la opción `multinomial` solo es compatible con los solvers ‘lbfgs’, ‘sag’, ‘saga’ y ‘newton-cg’).

> 🎓 El 'esquema' aquí puede ser 'ovr' (uno-contra-resto) o 'multinomial'. Dado que la regresión logística está realmente diseñada para soportar clasificación binaria, estos esquemas le permiten manejar mejor tareas de clasificación multicategoría. [fuente](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 El 'solver' se define como "el algoritmo a usar en el problema de optimización". [fuente](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ofrece esta tabla para explicar cómo los solvers manejan diferentes desafíos presentados por diferentes tipos de estructuras de datos:

![solvers](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/solvers.png)

## Ejercicio - dividir los datos

Podemos centrarnos en la regresión logística para nuestra primera prueba de entrenamiento, ya que recientemente aprendiste sobre esta en una lección anterior.
Divide tus datos en grupos de entrenamiento y prueba llamando a `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Ejercicio - aplicar regresión logística

Dado que estás usando el caso multicategoría, necesitas elegir qué _esquema_ usar y qué _solver_ configurar. Usa LogisticRegression con una configuración multicategoría y el solver **liblinear** para entrenar.

1. Crea una regresión logística con multi_class configurado como `ovr` y el solver configurado como `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prueba un solver diferente como `lbfgs`, que a menudo se configura como predeterminado.
> Nota, utiliza la función [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) de Pandas para aplanar tus datos cuando sea necesario.
¡La precisión es buena, con más del **80%**!

1. Puedes ver este modelo en acción probando una fila de datos (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    El resultado se imprime:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prueba con un número de fila diferente y verifica los resultados.

1. Profundizando más, puedes comprobar la precisión de esta predicción:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    El resultado se imprime: la cocina india es su mejor suposición, con buena probabilidad:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ ¿Puedes explicar por qué el modelo está bastante seguro de que se trata de una cocina india?

1. Obtén más detalles imprimiendo un informe de clasificación, como hiciste en las lecciones de regresión:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precisión | recall | f1-score | soporte |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | exactitud    | 0.80      | 1199   |          |         |
    | promedio macro | 0.80    | 0.80   | 0.80     | 1199    |
    | promedio ponderado | 0.80 | 0.80 | 0.80     | 1199    |

## 🚀Desafío

En esta lección, utilizaste tus datos limpios para construir un modelo de aprendizaje automático que puede predecir una cocina nacional basada en una serie de ingredientes. Tómate un tiempo para leer las muchas opciones que Scikit-learn ofrece para clasificar datos. Profundiza en el concepto de 'solver' para entender qué sucede detrás de escena.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Investiga un poco más sobre las matemáticas detrás de la regresión logística en [esta lección](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tarea 

[Estudia los solvers](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-04T22:24:07+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "es"
}
-->
# Clasificadores de cocina 2

En esta segunda lección de clasificación, explorarás más formas de clasificar datos numéricos. También aprenderás sobre las implicaciones de elegir un clasificador sobre otro.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

### Prerrequisitos

Asumimos que has completado las lecciones anteriores y tienes un conjunto de datos limpio en tu carpeta `data` llamado _cleaned_cuisines.csv_ en la raíz de esta carpeta de 4 lecciones.

### Preparación

Hemos cargado tu archivo _notebook.ipynb_ con el conjunto de datos limpio y lo hemos dividido en los dataframes X e y, listos para el proceso de construcción del modelo.

## Un mapa de clasificación

Anteriormente, aprendiste sobre las diversas opciones que tienes al clasificar datos utilizando la hoja de referencia de Microsoft. Scikit-learn ofrece una hoja de referencia similar, pero más detallada, que puede ayudarte aún más a reducir tus estimadores (otro término para clasificadores):

![Mapa de ML de Scikit-learn](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/3-Classifiers-2/images/map.png)
> Consejo: [visita este mapa en línea](https://scikit-learn.org/stable/tutorial/machine_learning_map/) y haz clic en el camino para leer la documentación.

### El plan

Este mapa es muy útil una vez que tienes una comprensión clara de tus datos, ya que puedes "caminar" por sus caminos hacia una decisión:

- Tenemos >50 muestras
- Queremos predecir una categoría
- Tenemos datos etiquetados
- Tenemos menos de 100K muestras
- ✨ Podemos elegir un Linear SVC
- Si eso no funciona, dado que tenemos datos numéricos
    - Podemos intentar un ✨ KNeighbors Classifier 
      - Si eso no funciona, probar ✨ SVC y ✨ Ensemble Classifiers

Este es un camino muy útil a seguir.

## Ejercicio - dividir los datos

Siguiendo este camino, deberíamos comenzar importando algunas bibliotecas para usar.

1. Importa las bibliotecas necesarias:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Divide tus datos de entrenamiento y prueba:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Clasificador Linear SVC

El clustering de vectores de soporte (SVC) es un miembro de la familia de técnicas de ML de máquinas de vectores de soporte (aprende más sobre estas abajo). En este método, puedes elegir un 'kernel' para decidir cómo agrupar las etiquetas. El parámetro 'C' se refiere a la 'regularización', que regula la influencia de los parámetros. El kernel puede ser uno de [varios](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); aquí lo configuramos como 'linear' para asegurarnos de aprovechar el Linear SVC. La probabilidad por defecto es 'false'; aquí la configuramos como 'true' para obtener estimaciones de probabilidad. Configuramos el estado aleatorio en '0' para mezclar los datos y obtener probabilidades.

### Ejercicio - aplicar un Linear SVC

Comienza creando un array de clasificadores. Irás añadiendo progresivamente a este array mientras probamos.

1. Comienza con un Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Entrena tu modelo usando el Linear SVC e imprime un informe:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    El resultado es bastante bueno:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Clasificador K-Neighbors

K-Neighbors es parte de la familia de métodos de ML "neighbors", que pueden ser utilizados tanto para aprendizaje supervisado como no supervisado. En este método, se crea un número predefinido de puntos y se recopilan datos alrededor de estos puntos para que se puedan predecir etiquetas generalizadas para los datos.

### Ejercicio - aplicar el clasificador K-Neighbors

El clasificador anterior fue bueno y funcionó bien con los datos, pero tal vez podamos obtener mejor precisión. Prueba un clasificador K-Neighbors.

1. Añade una línea a tu array de clasificadores (añade una coma después del elemento Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    El resultado es un poco peor:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Aprende sobre [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Clasificador de vectores de soporte

Los clasificadores de vectores de soporte son parte de la familia de [máquinas de vectores de soporte](https://wikipedia.org/wiki/Support-vector_machine) de métodos de ML que se utilizan para tareas de clasificación y regresión. Las SVM "mapean ejemplos de entrenamiento a puntos en el espacio" para maximizar la distancia entre dos categorías. Los datos posteriores se mapean en este espacio para que se pueda predecir su categoría.

### Ejercicio - aplicar un clasificador de vectores de soporte

Intentemos obtener una mejor precisión con un clasificador de vectores de soporte.

1. Añade una coma después del elemento K-Neighbors y luego añade esta línea:

    ```python
    'SVC': SVC(),
    ```

    ¡El resultado es bastante bueno!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Aprende sobre [vectores de soporte](https://scikit-learn.org/stable/modules/svm.html#svm)

## Clasificadores Ensemble

Sigamos el camino hasta el final, aunque la prueba anterior fue bastante buena. Probemos algunos clasificadores Ensemble, específicamente Random Forest y AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

El resultado es muy bueno, especialmente para Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Aprende sobre [clasificadores Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Este método de aprendizaje automático "combina las predicciones de varios estimadores base" para mejorar la calidad del modelo. En nuestro ejemplo, utilizamos Random Trees y AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), un método de promediación, construye un 'bosque' de 'árboles de decisión' infundidos con aleatoriedad para evitar el sobreajuste. El parámetro n_estimators se establece en el número de árboles.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajusta un clasificador a un conjunto de datos y luego ajusta copias de ese clasificador al mismo conjunto de datos. Se enfoca en los pesos de los elementos clasificados incorrectamente y ajusta el ajuste para el siguiente clasificador para corregir.

---

## 🚀Desafío

Cada una de estas técnicas tiene un gran número de parámetros que puedes ajustar. Investiga los parámetros predeterminados de cada una y piensa en lo que significaría ajustar estos parámetros para la calidad del modelo.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Hay mucho vocabulario técnico en estas lecciones, así que tómate un momento para revisar [esta lista](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) de terminología útil.

## Tarea 

[Prueba de parámetros](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-04T22:23:45+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "es"
}
-->
# Construir una Aplicación Web de Recomendación de Cocina

En esta lección, construirás un modelo de clasificación utilizando algunas de las técnicas que has aprendido en lecciones anteriores y con el delicioso conjunto de datos de cocina utilizado a lo largo de esta serie. Además, crearás una pequeña aplicación web para usar un modelo guardado, aprovechando el runtime web de Onnx.

Uno de los usos prácticos más útiles del aprendizaje automático es construir sistemas de recomendación, ¡y hoy puedes dar el primer paso en esa dirección!

[![Presentando esta aplicación web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "ML Aplicado")

> 🎥 Haz clic en la imagen de arriba para ver un video: Jen Looper construye una aplicación web utilizando datos clasificados de cocina.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

En esta lección aprenderás:

- Cómo construir un modelo y guardarlo como un modelo Onnx.
- Cómo usar Netron para inspeccionar el modelo.
- Cómo usar tu modelo en una aplicación web para inferencia.

## Construye tu modelo

Construir sistemas de aprendizaje automático aplicado es una parte importante de aprovechar estas tecnologías para tus sistemas empresariales. Puedes usar modelos dentro de tus aplicaciones web (y así utilizarlos en un contexto offline si es necesario) usando Onnx.

En una [lección anterior](../../3-Web-App/1-Web-App/README.md), construiste un modelo de Regresión sobre avistamientos de OVNIs, lo "encurtiste" y lo usaste en una aplicación Flask. Aunque esta arquitectura es muy útil de conocer, es una aplicación Python de pila completa, y tus requisitos pueden incluir el uso de una aplicación JavaScript.

En esta lección, puedes construir un sistema básico basado en JavaScript para inferencia. Sin embargo, primero necesitas entrenar un modelo y convertirlo para usarlo con Onnx.

## Ejercicio - entrenar un modelo de clasificación

Primero, entrena un modelo de clasificación utilizando el conjunto de datos de cocina limpio que usamos.

1. Comienza importando bibliotecas útiles:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Necesitas '[skl2onnx](https://onnx.ai/sklearn-onnx/)' para ayudar a convertir tu modelo de Scikit-learn al formato Onnx.

1. Luego, trabaja con tus datos de la misma manera que lo hiciste en lecciones anteriores, leyendo un archivo CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Elimina las dos primeras columnas innecesarias y guarda los datos restantes como 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Guarda las etiquetas como 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Comienza la rutina de entrenamiento

Usaremos la biblioteca 'SVC', que tiene buena precisión.

1. Importa las bibliotecas apropiadas de Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separa los conjuntos de entrenamiento y prueba:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Construye un modelo de Clasificación SVC como lo hiciste en la lección anterior:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Ahora, prueba tu modelo llamando a `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Imprime un informe de clasificación para verificar la calidad del modelo:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Como vimos antes, la precisión es buena:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Convierte tu modelo a Onnx

Asegúrate de hacer la conversión con el número adecuado de tensores. Este conjunto de datos tiene 380 ingredientes listados, por lo que necesitas anotar ese número en `FloatTensorType`:

1. Convierte usando un número de tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crea el archivo onx y guárdalo como **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, puedes pasar [opciones](https://onnx.ai/sklearn-onnx/parameterized.html) en tu script de conversión. En este caso, pasamos 'nocl' como True y 'zipmap' como False. Dado que este es un modelo de clasificación, tienes la opción de eliminar ZipMap, que produce una lista de diccionarios (no es necesario). `nocl` se refiere a la información de clase incluida en el modelo. Reduce el tamaño de tu modelo configurando `nocl` como 'True'.

Ejecutar todo el notebook ahora construirá un modelo Onnx y lo guardará en esta carpeta.

## Visualiza tu modelo

Los modelos Onnx no son muy visibles en Visual Studio Code, pero hay un software gratuito muy bueno que muchos investigadores usan para visualizar el modelo y asegurarse de que esté correctamente construido. Descarga [Netron](https://github.com/lutzroeder/Netron) y abre tu archivo model.onnx. Puedes ver tu modelo simple visualizado, con sus 380 entradas y clasificador listados:

![Visualización de Netron](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/4-Applied/images/netron.png)

Netron es una herramienta útil para visualizar tus modelos.

Ahora estás listo para usar este modelo en una aplicación web. Construyamos una aplicación que será útil cuando mires en tu refrigerador y trates de averiguar qué combinación de tus ingredientes sobrantes puedes usar para cocinar un plato determinado, según lo determine tu modelo.

## Construye una aplicación web de recomendación

Puedes usar tu modelo directamente en una aplicación web. Esta arquitectura también te permite ejecutarla localmente e incluso offline si es necesario. Comienza creando un archivo `index.html` en la misma carpeta donde guardaste tu archivo `model.onnx`.

1. En este archivo _index.html_, agrega el siguiente marcado:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Ahora, trabajando dentro de las etiquetas `body`, agrega un poco de marcado para mostrar una lista de casillas de verificación que reflejen algunos ingredientes:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Nota que a cada casilla de verificación se le asigna un valor. Esto refleja el índice donde se encuentra el ingrediente según el conjunto de datos. Por ejemplo, la manzana, en esta lista alfabética, ocupa la quinta columna, por lo que su valor es '4' ya que comenzamos a contar desde 0. Puedes consultar la [hoja de cálculo de ingredientes](../../../../4-Classification/data/ingredient_indexes.csv) para descubrir el índice de un ingrediente dado.

    Continuando tu trabajo en el archivo index.html, agrega un bloque de script donde se llame al modelo después del cierre final de `</div>`.

1. Primero, importa el [Runtime de Onnx](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > El Runtime de Onnx se utiliza para habilitar la ejecución de tus modelos Onnx en una amplia gama de plataformas de hardware, incluyendo optimizaciones y una API para usar.

1. Una vez que el Runtime esté en su lugar, puedes llamarlo:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

En este código, están ocurriendo varias cosas:

1. Creaste un array de 380 posibles valores (1 o 0) que se configurarán y enviarán al modelo para inferencia, dependiendo de si se marca una casilla de verificación de ingrediente.
2. Creaste un array de casillas de verificación y una forma de determinar si fueron marcadas en una función `init` que se llama cuando la aplicación comienza. Cuando se marca una casilla, el array `ingredients` se altera para reflejar el ingrediente elegido.
3. Creaste una función `testCheckboxes` que verifica si alguna casilla fue marcada.
4. Usas la función `startInference` cuando se presiona el botón y, si alguna casilla está marcada, comienzas la inferencia.
5. La rutina de inferencia incluye:
   1. Configurar una carga asincrónica del modelo.
   2. Crear una estructura de Tensor para enviar al modelo.
   3. Crear 'feeds' que reflejan la entrada `float_input` que creaste al entrenar tu modelo (puedes usar Netron para verificar ese nombre).
   4. Enviar estos 'feeds' al modelo y esperar una respuesta.

## Prueba tu aplicación

Abre una sesión de terminal en Visual Studio Code en la carpeta donde reside tu archivo index.html. Asegúrate de tener [http-server](https://www.npmjs.com/package/http-server) instalado globalmente y escribe `http-server` en el prompt. Debería abrirse un localhost y podrás ver tu aplicación web. Verifica qué cocina se recomienda según varios ingredientes:

![Aplicación web de ingredientes](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/4-Applied/images/web-app.png)

¡Felicidades, has creado una aplicación web de 'recomendación' con algunos campos! Tómate un tiempo para desarrollar este sistema.

## 🚀Desafío

Tu aplicación web es muy básica, así que continúa desarrollándola utilizando ingredientes y sus índices del archivo de datos [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). ¿Qué combinaciones de sabores funcionan para crear un plato nacional dado?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y Estudio Personal

Aunque esta lección solo tocó la utilidad de crear un sistema de recomendación para ingredientes de comida, esta área de aplicaciones de aprendizaje automático es muy rica en ejemplos. Lee más sobre cómo se construyen estos sistemas:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tarea 

[Construye un nuevo recomendador](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---


# Clustering

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:56:08+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "es"
}
-->
# Modelos de agrupamiento para aprendizaje automático

El agrupamiento es una tarea de aprendizaje automático que busca encontrar objetos que se asemejen entre sí y agruparlos en grupos llamados clústeres. Lo que diferencia el agrupamiento de otros enfoques en el aprendizaje automático es que todo sucede automáticamente; de hecho, es justo decir que es lo opuesto al aprendizaje supervisado.

## Tema regional: modelos de agrupamiento para los gustos musicales de una audiencia nigeriana 🎧

La diversa audiencia de Nigeria tiene gustos musicales variados. Usando datos extraídos de Spotify (inspirados en [este artículo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), analicemos algo de la música popular en Nigeria. Este conjunto de datos incluye información sobre el puntaje de 'bailabilidad', 'acústica', volumen, 'hablabilidad', popularidad y energía de varias canciones. ¡Será interesante descubrir patrones en estos datos!

![Un tocadiscos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/turntable.f2b86b13c53302dc.webp)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> en <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
En esta serie de lecciones, descubrirás nuevas formas de analizar datos utilizando técnicas de agrupamiento. El agrupamiento es particularmente útil cuando tu conjunto de datos carece de etiquetas. Si tiene etiquetas, entonces las técnicas de clasificación como las que aprendiste en lecciones anteriores podrían ser más útiles. Pero en casos donde buscas agrupar datos sin etiquetar, el agrupamiento es una excelente manera de descubrir patrones.

> Hay herramientas útiles de bajo código que pueden ayudarte a aprender a trabajar con modelos de agrupamiento. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecciones

1. [Introducción al agrupamiento](1-Visualize/README.md)
2. [Agrupamiento K-Means](2-K-Means/README.md)

## Créditos

Estas lecciones fueron escritas con 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) con revisiones útiles de [Rishit Dagli](https://rishit_dagli) y [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

El conjunto de datos [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) fue obtenido de Kaggle como datos extraídos de Spotify.

Ejemplos útiles de K-Means que ayudaron a crear esta lección incluyen esta [exploración de iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introductorio](https://www.kaggle.com/prashant111/k-means-clustering-with-python), y este [ejemplo hipotético de una ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-04T22:17:37+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "es"
}
-->
# Introducción a la agrupación

La agrupación es un tipo de [aprendizaje no supervisado](https://wikipedia.org/wiki/Unsupervised_learning) que asume que un conjunto de datos no está etiquetado o que sus entradas no están asociadas con salidas predefinidas. Utiliza varios algoritmos para clasificar datos no etiquetados y proporcionar agrupaciones según los patrones que detecta en los datos.

[![No One Like You de PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You de PSquare")

> 🎥 Haz clic en la imagen de arriba para ver un video. Mientras estudias aprendizaje automático con agrupación, disfruta de algunos temas de Dance Hall nigeriano: esta es una canción muy popular de 2014 de PSquare.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

### Introducción

[La agrupación](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) es muy útil para la exploración de datos. Veamos si puede ayudar a descubrir tendencias y patrones en la forma en que las audiencias nigerianas consumen música.

✅ Tómate un minuto para pensar en los usos de la agrupación. En la vida real, la agrupación ocurre cada vez que tienes un montón de ropa y necesitas clasificar la ropa de los miembros de tu familia 🧦👕👖🩲. En ciencia de datos, la agrupación ocurre al intentar analizar las preferencias de un usuario o determinar las características de cualquier conjunto de datos no etiquetado. La agrupación, de alguna manera, ayuda a dar sentido al caos, como un cajón de calcetines.

[![Introducción al aprendizaje automático](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introducción a la agrupación")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT introduce la agrupación.

En un entorno profesional, la agrupación puede usarse para determinar cosas como la segmentación de mercado, identificando qué grupos de edad compran qué productos, por ejemplo. Otro uso sería la detección de anomalías, tal vez para identificar fraudes en un conjunto de datos de transacciones con tarjetas de crédito. O podrías usar la agrupación para identificar tumores en un lote de escaneos médicos.

✅ Piensa un minuto en cómo podrías haber encontrado la agrupación 'en la vida real', en un entorno bancario, de comercio electrónico o empresarial.

> 🎓 Curiosamente, el análisis de agrupación se originó en los campos de la Antropología y la Psicología en la década de 1930. ¿Puedes imaginar cómo podría haberse utilizado?

Alternativamente, podrías usarlo para agrupar resultados de búsqueda, como enlaces de compras, imágenes o reseñas, por ejemplo. La agrupación es útil cuando tienes un conjunto de datos grande que deseas reducir y sobre el cual deseas realizar un análisis más detallado, por lo que la técnica puede usarse para aprender sobre los datos antes de construir otros modelos.

✅ Una vez que tus datos están organizados en grupos, les asignas un Id de grupo, y esta técnica puede ser útil para preservar la privacidad de un conjunto de datos; en lugar de referirte a un punto de datos por información identificable, puedes referirte a él por su Id de grupo. ¿Puedes pensar en otras razones por las que preferirías referirte a un Id de grupo en lugar de otros elementos del grupo para identificarlo?

Profundiza tu comprensión de las técnicas de agrupación en este [módulo de aprendizaje](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Introducción a la agrupación

[Scikit-learn ofrece una amplia variedad](https://scikit-learn.org/stable/modules/clustering.html) de métodos para realizar agrupación. El tipo que elijas dependerá de tu caso de uso. Según la documentación, cada método tiene varios beneficios. Aquí hay una tabla simplificada de los métodos compatibles con Scikit-learn y sus casos de uso apropiados:

| Nombre del método            | Caso de uso                                                            |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | propósito general, inductivo                                           |
| Affinity propagation         | muchos, grupos desiguales, inductivo                                  |
| Mean-shift                   | muchos, grupos desiguales, inductivo                                  |
| Spectral clustering          | pocos, grupos iguales, transductivo                                   |
| Ward hierarchical clustering | muchos, grupos restringidos, transductivo                             |
| Agglomerative clustering     | muchos, restringidos, distancias no euclidianas, transductivo         |
| DBSCAN                       | geometría no plana, grupos desiguales, transductivo                   |
| OPTICS                       | geometría no plana, grupos desiguales con densidad variable, transductivo |
| Gaussian mixtures            | geometría plana, inductivo                                            |
| BIRCH                        | conjunto de datos grande con valores atípicos, inductivo              |

> 🎓 Cómo creamos grupos tiene mucho que ver con cómo agrupamos los puntos de datos. Desglosémoslo:
>
> 🎓 ['Transductivo' vs. 'inductivo'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> La inferencia transductiva se deriva de casos de entrenamiento observados que se asignan a casos de prueba específicos. La inferencia inductiva se deriva de casos de entrenamiento que se asignan a reglas generales que luego se aplican a casos de prueba.
> 
> Un ejemplo: Imagina que tienes un conjunto de datos que está solo parcialmente etiquetado. Algunas cosas son 'discos', otras 'CDs', y otras están en blanco. Tu trabajo es proporcionar etiquetas para los elementos en blanco. Si eliges un enfoque inductivo, entrenarías un modelo buscando 'discos' y 'CDs', y aplicarías esas etiquetas a tus datos no etiquetados. Este enfoque tendrá problemas para clasificar cosas que en realidad son 'cassettes'. Un enfoque transductivo, por otro lado, maneja estos datos desconocidos de manera más efectiva al agrupar elementos similares y luego aplicar una etiqueta a un grupo. En este caso, los grupos podrían reflejar 'cosas musicales redondas' y 'cosas musicales cuadradas'.
> 
> 🎓 ['Geometría no plana' vs. 'plana'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivado de la terminología matemática, la geometría no plana vs. plana se refiere a la medida de distancias entre puntos mediante métodos geométricos 'planos' ([Euclidianos](https://wikipedia.org/wiki/Euclidean_geometry)) o 'no planos' (no Euclidianos).
>
>'Plana' en este contexto se refiere a la geometría Euclidiana (partes de la cual se enseñan como geometría 'plana'), y no plana se refiere a la geometría no Euclidiana. ¿Qué tiene que ver la geometría con el aprendizaje automático? Bueno, como dos campos que están arraigados en las matemáticas, debe haber una forma común de medir distancias entre puntos en grupos, y eso puede hacerse de manera 'plana' o 'no plana', dependiendo de la naturaleza de los datos. [Las distancias Euclidianas](https://wikipedia.org/wiki/Euclidean_distance) se miden como la longitud de un segmento de línea entre dos puntos. [Las distancias no Euclidianas](https://wikipedia.org/wiki/Non-Euclidean_geometry) se miden a lo largo de una curva. Si tus datos, visualizados, parecen no existir en un plano, podrías necesitar usar un algoritmo especializado para manejarlos.
>
![Infografía de geometría plana vs. no plana](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distancias'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Los grupos se definen por su matriz de distancias, es decir, las distancias entre puntos. Esta distancia puede medirse de varias maneras. Los grupos Euclidianos se definen por el promedio de los valores de los puntos y contienen un 'centroide' o punto central. Las distancias se miden así por la distancia a ese centroide. Las distancias no Euclidianas se refieren a 'clustroides', el punto más cercano a otros puntos. Los clustroides, a su vez, pueden definirse de varias maneras.
> 
> 🎓 ['Restringido'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [La agrupación restringida](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduce el aprendizaje 'semi-supervisado' en este método no supervisado. Las relaciones entre puntos se marcan como 'no puede vincular' o 'debe vincular', por lo que se imponen algunas reglas al conjunto de datos.
>
>Un ejemplo: Si un algoritmo se deja libre en un lote de datos no etiquetados o semi-etiquetados, los grupos que produce pueden ser de baja calidad. En el ejemplo anterior, los grupos podrían agrupar 'cosas musicales redondas', 'cosas musicales cuadradas', 'cosas triangulares' y 'galletas'. Si se le dan algunas restricciones o reglas a seguir ("el artículo debe estar hecho de plástico", "el artículo necesita poder producir música"), esto puede ayudar a 'restringir' el algoritmo para tomar mejores decisiones.
> 
> 🎓 'Densidad'
> 
> Los datos que son 'ruidosos' se consideran 'densos'. Las distancias entre puntos en cada uno de sus grupos pueden resultar, al examinarlas, más o menos densas, o 'congestionadas', y por lo tanto estos datos necesitan analizarse con el método de agrupación apropiado. [Este artículo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demuestra la diferencia entre usar agrupación K-Means vs. algoritmos HDBSCAN para explorar un conjunto de datos ruidoso con densidad de grupo desigual.

## Algoritmos de agrupación

Existen más de 100 algoritmos de agrupación, y su uso depende de la naturaleza de los datos en cuestión. Hablemos de algunos de los principales:

- **Agrupación jerárquica**. Si un objeto se clasifica por su proximidad a un objeto cercano, en lugar de a uno más lejano, los grupos se forman en función de la distancia de sus miembros hacia y desde otros objetos. La agrupación aglomerativa de Scikit-learn es jerárquica.

   ![Infografía de agrupación jerárquica](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Agrupación por centroides**. Este algoritmo popular requiere elegir 'k', o el número de grupos a formar, después de lo cual el algoritmo determina el punto central de un grupo y reúne datos alrededor de ese punto. [La agrupación K-means](https://wikipedia.org/wiki/K-means_clustering) es una versión popular de la agrupación por centroides. El centro se determina por la media más cercana, de ahí el nombre. La distancia cuadrada desde el grupo se minimiza.

   ![Infografía de agrupación por centroides](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/centroid.png)
   > Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Agrupación basada en distribución**. Basada en modelos estadísticos, la agrupación basada en distribución se centra en determinar la probabilidad de que un punto de datos pertenezca a un grupo y asignarlo en consecuencia. Los métodos de mezcla gaussiana pertenecen a este tipo.

- **Agrupación basada en densidad**. Los puntos de datos se asignan a grupos según su densidad, o su agrupación entre sí. Los puntos de datos alejados del grupo se consideran valores atípicos o ruido. DBSCAN, Mean-shift y OPTICS pertenecen a este tipo de agrupación.

- **Agrupación basada en cuadrícula**. Para conjuntos de datos multidimensionales, se crea una cuadrícula y los datos se dividen entre las celdas de la cuadrícula, creando así grupos.

## Ejercicio - agrupa tus datos

La técnica de agrupación se beneficia enormemente de una visualización adecuada, así que comencemos visualizando nuestros datos musicales. Este ejercicio nos ayudará a decidir cuál de los métodos de agrupación deberíamos usar de manera más efectiva para la naturaleza de estos datos.

1. Abre el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) en esta carpeta.

1. Importa el paquete `Seaborn` para una buena visualización de datos.

    ```python
    !pip install seaborn
    ```

1. Agrega los datos de canciones desde [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Carga un dataframe con algunos datos sobre las canciones. Prepárate para explorar estos datos importando las bibliotecas y mostrando los datos:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Revisa las primeras líneas de datos:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtén información sobre el dataframe llamando a `info()`:

    ```python
    df.info()
    ```

   El resultado se ve así:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Verifica nuevamente si hay valores nulos llamando a `isnull()` y asegurándote de que la suma sea 0:

    ```python
    df.isnull().sum()
    ```

    Todo se ve bien:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Describe los datos:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Si estamos trabajando con clustering, un método no supervisado que no requiere datos etiquetados, ¿por qué estamos mostrando estos datos con etiquetas? En la fase de exploración de datos son útiles, pero no son necesarios para que los algoritmos de clustering funcionen. Podrías eliminar los encabezados de las columnas y referirte a los datos por número de columna.

Observa los valores generales de los datos. Nota que la popularidad puede ser '0', lo que muestra canciones que no tienen ranking. Eliminemos esos valores pronto.

1. Usa un gráfico de barras para encontrar los géneros más populares:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/popular.png)

✅ Si deseas ver más valores principales, cambia el top `[:5]` a un valor mayor o elimínalo para ver todo.

Nota que cuando el género principal se describe como 'Missing', significa que Spotify no lo clasificó, así que eliminémoslo.

1. Elimina los datos faltantes filtrándolos:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Ahora verifica nuevamente los géneros:

    ![most popular](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/all-genres.png)

1. Los tres géneros principales dominan este conjunto de datos. Concentrémonos en `afro dancehall`, `afropop` y `nigerian pop`, y además filtremos el conjunto de datos para eliminar cualquier valor de popularidad igual a 0 (lo que significa que no fue clasificado con una popularidad en el conjunto de datos y puede considerarse ruido para nuestros propósitos):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Haz una prueba rápida para ver si los datos tienen alguna correlación particularmente fuerte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/correlation.png)

    La única correlación fuerte es entre `energy` y `loudness`, lo cual no es muy sorprendente, dado que la música fuerte suele ser bastante energética. Por lo demás, las correlaciones son relativamente débiles. Será interesante ver qué puede hacer un algoritmo de clustering con estos datos.

    > 🎓 ¡Nota que la correlación no implica causalidad! Tenemos prueba de correlación pero no prueba de causalidad. Un [sitio web divertido](https://tylervigen.com/spurious-correlations) tiene algunos gráficos que enfatizan este punto.

¿Hay alguna convergencia en este conjunto de datos en torno a la popularidad percibida de una canción y su capacidad de baile? Un FacetGrid muestra que hay círculos concéntricos que se alinean, independientemente del género. ¿Podría ser que los gustos nigerianos convergen en un cierto nivel de capacidad de baile para este género?

✅ Prueba diferentes puntos de datos (energy, loudness, speechiness) y más o diferentes géneros musicales. ¿Qué puedes descubrir? Mira la tabla `df.describe()` para ver la distribución general de los puntos de datos.

### Ejercicio - distribución de datos

¿Son estos tres géneros significativamente diferentes en la percepción de su capacidad de baile, basada en su popularidad?

1. Examina la distribución de datos de nuestros tres géneros principales para popularidad y capacidad de baile a lo largo de un eje x y y dado.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Puedes descubrir círculos concéntricos alrededor de un punto general de convergencia, mostrando la distribución de puntos.

    > 🎓 Nota que este ejemplo utiliza un gráfico KDE (Kernel Density Estimate) que representa los datos usando una curva de densidad de probabilidad continua. Esto nos permite interpretar los datos cuando trabajamos con múltiples distribuciones.

    En general, los tres géneros se alinean de manera suelta en términos de su popularidad y capacidad de baile. Determinar clusters en estos datos alineados de manera suelta será un desafío:

    ![distribution](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/distribution.png)

1. Crea un gráfico de dispersión:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un gráfico de dispersión de los mismos ejes muestra un patrón similar de convergencia.

    ![Facetgrid](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/images/facetgrid.png)

En general, para clustering, puedes usar gráficos de dispersión para mostrar clusters de datos, por lo que dominar este tipo de visualización es muy útil. En la próxima lección, tomaremos estos datos filtrados y usaremos clustering k-means para descubrir grupos en estos datos que parecen superponerse de maneras interesantes.

---

## 🚀Desafío

En preparación para la próxima lección, haz un gráfico sobre los diversos algoritmos de clustering que podrías descubrir y usar en un entorno de producción. ¿Qué tipo de problemas está tratando de abordar el clustering?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Antes de aplicar algoritmos de clustering, como hemos aprendido, es una buena idea entender la naturaleza de tu conjunto de datos. Lee más sobre este tema [aquí](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Este artículo útil](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) te guía a través de las diferentes formas en que varios algoritmos de clustering se comportan, dadas diferentes formas de datos.

## Tarea

[Investiga otras visualizaciones para clustering](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-04T22:18:39+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "es"
}
-->
# Agrupamiento K-Means

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

En esta lección, aprenderás a crear agrupaciones utilizando Scikit-learn y el conjunto de datos de música nigeriana que importaste anteriormente. Cubriremos los conceptos básicos de K-Means para el agrupamiento. Ten en cuenta que, como aprendiste en la lección anterior, hay muchas formas de trabajar con agrupaciones y el método que utilices dependerá de tus datos. Probaremos K-Means, ya que es la técnica de agrupamiento más común. ¡Comencemos!

Términos que aprenderás:

- Puntuación de silueta
- Método del codo
- Inercia
- Varianza

## Introducción

[El agrupamiento K-Means](https://wikipedia.org/wiki/K-means_clustering) es un método derivado del ámbito del procesamiento de señales. Se utiliza para dividir y particionar grupos de datos en 'k' agrupaciones utilizando una serie de observaciones. Cada observación trabaja para agrupar un punto de datos dado al 'promedio' más cercano, o el punto central de una agrupación.

Las agrupaciones pueden visualizarse como [diagramas de Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), que incluyen un punto (o 'semilla') y su región correspondiente.

![diagrama de voronoi](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/images/voronoi.png)

> Infografía por [Jen Looper](https://twitter.com/jenlooper)

El proceso de agrupamiento K-Means [se ejecuta en tres pasos](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. El algoritmo selecciona un número k de puntos centrales muestreando del conjunto de datos. Después, entra en un bucle:
    1. Asigna cada muestra al centroide más cercano.
    2. Crea nuevos centroides tomando el valor promedio de todas las muestras asignadas a los centroides anteriores.
    3. Luego, calcula la diferencia entre los nuevos y antiguos centroides y repite hasta que los centroides se estabilicen.

Una desventaja de usar K-Means es que necesitas establecer 'k', es decir, el número de centroides. Afortunadamente, el 'método del codo' ayuda a estimar un buen valor inicial para 'k'. Lo probarás en un momento.

## Prerrequisito

Trabajarás en el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) de esta lección, que incluye la importación de datos y la limpieza preliminar que realizaste en la lección anterior.

## Ejercicio - preparación

Comienza revisando nuevamente los datos de las canciones.

1. Crea un diagrama de caja llamando a `boxplot()` para cada columna:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Estos datos son un poco ruidosos: al observar cada columna como un diagrama de caja, puedes ver valores atípicos.

    ![valores atípicos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/images/boxplots.png)

Podrías recorrer el conjunto de datos y eliminar estos valores atípicos, pero eso haría que los datos sean bastante mínimos.

1. Por ahora, elige qué columnas usarás para tu ejercicio de agrupamiento. Escoge aquellas con rangos similares y codifica la columna `artist_top_genre` como datos numéricos:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Ahora necesitas elegir cuántas agrupaciones apuntar. Sabes que hay 3 géneros musicales que extrajimos del conjunto de datos, así que probemos con 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Verás un arreglo impreso con las agrupaciones predichas (0, 1 o 2) para cada fila del dataframe.

1. Usa este arreglo para calcular una 'puntuación de silueta':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Puntuación de silueta

Busca una puntuación de silueta cercana a 1. Esta puntuación varía de -1 a 1, y si la puntuación es 1, la agrupación es densa y está bien separada de otras agrupaciones. Un valor cercano a 0 representa agrupaciones superpuestas con muestras muy cercanas al límite de decisión de las agrupaciones vecinas. [(Fuente)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Nuestra puntuación es **0.53**, justo en el medio. Esto indica que nuestros datos no son particularmente adecuados para este tipo de agrupamiento, pero sigamos adelante.

### Ejercicio - construir un modelo

1. Importa `KMeans` y comienza el proceso de agrupamiento.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Hay algunas partes aquí que merecen explicación.

    > 🎓 range: Estas son las iteraciones del proceso de agrupamiento.

    > 🎓 random_state: "Determina la generación de números aleatorios para la inicialización de centroides." [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "suma de cuadrados dentro de la agrupación" mide la distancia promedio al cuadrado de todos los puntos dentro de una agrupación al centroide de la agrupación. [Fuente](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inercia: Los algoritmos K-Means intentan elegir centroides para minimizar la 'inercia', "una medida de cuán coherentes son internamente las agrupaciones." [Fuente](https://scikit-learn.org/stable/modules/clustering.html). El valor se agrega a la variable wcss en cada iteración.

    > 🎓 k-means++: En [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) puedes usar la optimización 'k-means++', que "inicializa los centroides para que estén (generalmente) distantes entre sí, lo que lleva a resultados probablemente mejores que la inicialización aleatoria."

### Método del codo

Anteriormente, dedujiste que, dado que apuntaste a 3 géneros musicales, deberías elegir 3 agrupaciones. ¿Pero es ese el caso?

1. Usa el 'método del codo' para asegurarte.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Usa la variable `wcss` que construiste en el paso anterior para crear un gráfico que muestre dónde está el 'doblez' en el codo, lo que indica el número óptimo de agrupaciones. ¡Quizás sí sea 3!

    ![método del codo](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/images/elbow.png)

## Ejercicio - mostrar las agrupaciones

1. Intenta el proceso nuevamente, esta vez configurando tres agrupaciones, y muestra las agrupaciones como un diagrama de dispersión:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Verifica la precisión del modelo:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    La precisión de este modelo no es muy buena, y la forma de las agrupaciones te da una pista del porqué.

    ![agrupaciones](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/images/clusters.png)

    Estos datos están demasiado desequilibrados, poco correlacionados y hay demasiada varianza entre los valores de las columnas para agruparlos bien. De hecho, las agrupaciones que se forman probablemente están muy influenciadas o sesgadas por las tres categorías de géneros que definimos anteriormente. ¡Fue un proceso de aprendizaje!

    En la documentación de Scikit-learn, puedes ver que un modelo como este, con agrupaciones no muy bien delimitadas, tiene un problema de 'varianza':

    ![modelos problemáticos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/images/problems.png)
    > Infografía de Scikit-learn

## Varianza

La varianza se define como "el promedio de las diferencias al cuadrado respecto a la media" [(Fuente)](https://www.mathsisfun.com/data/standard-deviation.html). En el contexto de este problema de agrupamiento, se refiere a datos cuyos números tienden a divergir demasiado de la media.

✅ Este es un buen momento para pensar en todas las formas en que podrías corregir este problema. ¿Ajustar un poco más los datos? ¿Usar diferentes columnas? ¿Usar un algoritmo diferente? Pista: Prueba [escalar tus datos](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) para normalizarlos y prueba con otras columnas.

> Prueba este '[calculador de varianza](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' para entender mejor el concepto.

---

## 🚀Desafío

Dedica algo de tiempo a este notebook ajustando parámetros. ¿Puedes mejorar la precisión del modelo limpiando más los datos (eliminando valores atípicos, por ejemplo)? Puedes usar pesos para dar más importancia a ciertas muestras de datos. ¿Qué más puedes hacer para crear mejores agrupaciones?

Pista: Prueba escalar tus datos. Hay código comentado en el notebook que agrega escalado estándar para hacer que las columnas de datos se parezcan más en términos de rango. Descubrirás que, aunque la puntuación de silueta disminuye, el 'doblez' en el gráfico del codo se suaviza. Esto se debe a que dejar los datos sin escalar permite que los datos con menos varianza tengan más peso. Lee un poco más sobre este problema [aquí](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y autoestudio

Echa un vistazo a un simulador de K-Means [como este](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Puedes usar esta herramienta para visualizar puntos de datos de muestra y determinar sus centroides. Puedes editar la aleatoriedad de los datos, el número de agrupaciones y el número de centroides. ¿Te ayuda esto a tener una idea de cómo se pueden agrupar los datos?

Además, revisa [este documento sobre K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) de Stanford.

## Tarea

[Prueba diferentes métodos de agrupamiento](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---
