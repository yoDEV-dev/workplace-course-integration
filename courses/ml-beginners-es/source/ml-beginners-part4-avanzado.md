# ML para Principiantes 🤖
## Parte 4: Temas Avanzados
**Series de Tiempo, Aprendizaje por Refuerzo y Aplicaciones**

---


# Series de Tiempo

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61342603bad8acadbc6b2e4e3aab3f66",
  "translation_date": "2025-09-03T22:41:30+00:00",
  "source_file": "7-TimeSeries/README.md",
  "language_code": "es"
}
-->
# Introducción a la predicción de series temporales

¿Qué es la predicción de series temporales? Se trata de predecir eventos futuros analizando las tendencias del pasado.

## Tema regional: uso mundial de electricidad ✨

En estas dos lecciones, se te presentará la predicción de series temporales, un área de aprendizaje automático algo menos conocida pero que es extremadamente valiosa para aplicaciones industriales y empresariales, entre otros campos. Aunque las redes neuronales pueden usarse para mejorar la utilidad de estos modelos, los estudiaremos en el contexto del aprendizaje automático clásico, ya que los modelos ayudan a predecir el rendimiento futuro basándose en el pasado.

Nuestro enfoque regional es el uso eléctrico en el mundo, un conjunto de datos interesante para aprender sobre la predicción del consumo futuro de energía basado en patrones de carga pasados. Puedes ver cómo este tipo de predicción puede ser extremadamente útil en un entorno empresarial.

![red eléctrica](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/electric-grid.0c21d5214db09ffa.webp)

Foto de [Peddi Sai hrithik](https://unsplash.com/@shutter_log?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) de torres eléctricas en una carretera en Rajasthan en [Unsplash](https://unsplash.com/s/photos/electric-india?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## Lecciones

1. [Introducción a la predicción de series temporales](1-Introduction/README.md)
2. [Construcción de modelos ARIMA para series temporales](2-ARIMA/README.md)
3. [Construcción de un Support Vector Regressor para la predicción de series temporales](3-SVR/README.md)

## Créditos

"Introducción a la predicción de series temporales" fue escrito con ⚡️ por [Francesca Lazzeri](https://twitter.com/frlazzeri) y [Jen Looper](https://twitter.com/jenlooper). Los cuadernos aparecieron por primera vez en línea en el [repositorio de Azure "Deep Learning For Time Series"](https://github.com/Azure/DeepLearningForTimeSeriesForecasting) originalmente escrito por Francesca Lazzeri. La lección de SVR fue escrita por [Anirban Mukherjee](https://github.com/AnirbanMukherjeeXD).

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-04T22:15:29+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "es"
}
-->
# Introducción a la predicción de series temporales

![Resumen de series temporales en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-timeseries.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

En esta lección y la siguiente, aprenderás un poco sobre la predicción de series temporales, una parte interesante y valiosa del repertorio de un científico de ML que es un poco menos conocida que otros temas. La predicción de series temporales es como una especie de 'bola de cristal': basándote en el rendimiento pasado de una variable como el precio, puedes predecir su valor potencial futuro.

[![Introducción a la predicción de series temporales](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introducción a la predicción de series temporales")

> 🎥 Haz clic en la imagen de arriba para ver un video sobre la predicción de series temporales

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

Es un campo útil e interesante con un valor real para los negocios, dado su uso directo en problemas de precios, inventarios y cuestiones de la cadena de suministro. Aunque las técnicas de aprendizaje profundo han comenzado a usarse para obtener más información y predecir mejor el rendimiento futuro, la predicción de series temporales sigue siendo un campo muy influenciado por técnicas clásicas de ML.

> El útil currículo de series temporales de Penn State se puede encontrar [aquí](https://online.stat.psu.edu/stat510/lesson/1)

## Introducción

Supongamos que gestionas una red de parquímetros inteligentes que proporcionan datos sobre la frecuencia y duración de su uso a lo largo del tiempo.

> ¿Qué pasaría si pudieras predecir, basándote en el rendimiento pasado del parquímetro, su valor futuro según las leyes de la oferta y la demanda?

Predecir con precisión cuándo actuar para lograr tu objetivo es un desafío que podría abordarse con la predicción de series temporales. No haría feliz a la gente que se les cobre más en horas pico cuando buscan un lugar para estacionar, ¡pero sería una forma segura de generar ingresos para limpiar las calles!

Exploremos algunos de los tipos de algoritmos de series temporales y comencemos un cuaderno para limpiar y preparar algunos datos. Los datos que analizarás provienen de la competencia de predicción GEFCom2014. Consisten en 3 años de valores horarios de carga eléctrica y temperatura entre 2012 y 2014. Dado el patrón histórico de carga eléctrica y temperatura, puedes predecir valores futuros de carga eléctrica.

En este ejemplo, aprenderás a predecir un paso de tiempo hacia adelante, utilizando únicamente datos históricos de carga. Sin embargo, antes de comenzar, es útil entender qué está sucediendo detrás de escena.

## Algunas definiciones

Cuando te encuentres con el término 'series temporales', necesitas entender su uso en varios contextos diferentes.

🎓 **Series temporales**

En matemáticas, "una serie temporal es una serie de puntos de datos indexados (o listados o graficados) en orden temporal. Más comúnmente, una serie temporal es una secuencia tomada en puntos sucesivos igualmente espaciados en el tiempo". Un ejemplo de una serie temporal es el valor de cierre diario del [Promedio Industrial Dow Jones](https://wikipedia.org/wiki/Time_series). El uso de gráficos de series temporales y el modelado estadístico se encuentra frecuentemente en el procesamiento de señales, la predicción del clima, la predicción de terremotos y otros campos donde ocurren eventos y los puntos de datos pueden graficarse a lo largo del tiempo.

🎓 **Análisis de series temporales**

El análisis de series temporales es el análisis de los datos de series temporales mencionados anteriormente. Los datos de series temporales pueden tomar formas distintas, incluyendo 'series temporales interrumpidas', que detectan patrones en la evolución de una serie temporal antes y después de un evento interruptor. El tipo de análisis necesario para la serie temporal depende de la naturaleza de los datos. Los datos de series temporales en sí mismos pueden tomar la forma de series de números o caracteres.

El análisis que se realiza utiliza una variedad de métodos, incluidos dominio de frecuencia y dominio de tiempo, lineales y no lineales, y más. [Aprende más](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sobre las muchas formas de analizar este tipo de datos.

🎓 **Predicción de series temporales**

La predicción de series temporales es el uso de un modelo para predecir valores futuros basados en patrones mostrados por datos previamente recopilados a medida que ocurrieron en el pasado. Aunque es posible usar modelos de regresión para explorar datos de series temporales, con índices de tiempo como variables x en un gráfico, dichos datos se analizan mejor utilizando tipos especiales de modelos.

Los datos de series temporales son una lista de observaciones ordenadas, a diferencia de los datos que pueden analizarse mediante regresión lineal. El más común es ARIMA, un acrónimo que significa "Promedio Móvil Integrado Autorregresivo".

[Modelos ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relacionan el valor presente de una serie con valores pasados y errores de predicción pasados". Son más apropiados para analizar datos en el dominio del tiempo, donde los datos están ordenados a lo largo del tiempo.

> Hay varios tipos de modelos ARIMA, que puedes aprender [aquí](https://people.duke.edu/~rnau/411arim.htm) y que abordarás en la próxima lección.

En la próxima lección, construirás un modelo ARIMA utilizando [Series Temporales Univariadas](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), que se enfoca en una variable que cambia su valor a lo largo del tiempo. Un ejemplo de este tipo de datos es [este conjunto de datos](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) que registra la concentración mensual de CO2 en el Observatorio Mauna Loa:

|   CO2   | YearMonth | Year  | Month |
| :-----: | :-------: | :---: | :---: |
| 330.62  |  1975.04  | 1975  |   1   |
| 331.40  |  1975.13  | 1975  |   2   |
| 331.87  |  1975.21  | 1975  |   3   |
| 333.18  |  1975.29  | 1975  |   4   |
| 333.92  |  1975.38  | 1975  |   5   |
| 333.43  |  1975.46  | 1975  |   6   |
| 331.85  |  1975.54  | 1975  |   7   |
| 330.01  |  1975.63  | 1975  |   8   |
| 328.51  |  1975.71  | 1975  |   9   |
| 328.41  |  1975.79  | 1975  |  10   |
| 329.25  |  1975.88  | 1975  |  11   |
| 330.97  |  1975.96  | 1975  |  12   |

✅ Identifica la variable que cambia a lo largo del tiempo en este conjunto de datos.

## Características de los datos de series temporales a considerar

Al observar datos de series temporales, podrías notar que tienen [ciertas características](https://online.stat.psu.edu/stat510/lesson/1/1.1) que necesitas tener en cuenta y mitigar para comprender mejor sus patrones. Si consideras los datos de series temporales como un posible 'señal' que deseas analizar, estas características pueden considerarse 'ruido'. A menudo necesitarás reducir este 'ruido' compensando algunas de estas características utilizando técnicas estadísticas.

Aquí hay algunos conceptos que deberías conocer para trabajar con series temporales:

🎓 **Tendencias**

Las tendencias se definen como aumentos y disminuciones medibles a lo largo del tiempo. [Lee más](https://machinelearningmastery.com/time-series-trends-in-python). En el contexto de series temporales, se trata de cómo usar y, si es necesario, eliminar tendencias de tus series temporales.

🎓 **[Estacionalidad](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

La estacionalidad se define como fluctuaciones periódicas, como las compras navideñas que podrían afectar las ventas, por ejemplo. [Echa un vistazo](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) a cómo diferentes tipos de gráficos muestran la estacionalidad en los datos.

🎓 **Valores atípicos**

Los valores atípicos están lejos de la varianza estándar de los datos.

🎓 **Ciclo a largo plazo**

Independientemente de la estacionalidad, los datos podrían mostrar un ciclo a largo plazo, como una recesión económica que dura más de un año.

🎓 **Varianza constante**

Con el tiempo, algunos datos muestran fluctuaciones constantes, como el uso de energía durante el día y la noche.

🎓 **Cambios abruptos**

Los datos podrían mostrar un cambio abrupto que podría necesitar un análisis más profundo. El cierre repentino de negocios debido al COVID, por ejemplo, causó cambios en los datos.

✅ Aquí hay un [ejemplo de gráfico de series temporales](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) que muestra el gasto diario en moneda dentro del juego durante algunos años. ¿Puedes identificar alguna de las características mencionadas anteriormente en estos datos?

![Gasto en moneda dentro del juego](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/1-Introduction/images/currency.png)

## Ejercicio - comenzando con datos de uso de energía

Comencemos creando un modelo de series temporales para predecir el uso futuro de energía dado el uso pasado.

> Los datos en este ejemplo provienen de la competencia de predicción GEFCom2014. Consisten en 3 años de valores horarios de carga eléctrica y temperatura entre 2012 y 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli y Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, julio-septiembre, 2016.

1. En la carpeta `working` de esta lección, abre el archivo _notebook.ipynb_. Comienza agregando bibliotecas que te ayudarán a cargar y visualizar datos.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Nota: estás utilizando los archivos de la carpeta `common` incluida, que configuran tu entorno y manejan la descarga de los datos.

2. A continuación, examina los datos como un dataframe llamando a `load_data()` y `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Puedes ver que hay dos columnas que representan la fecha y la carga:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Ahora, grafica los datos llamando a `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![gráfico de energía](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Ahora, grafica la primera semana de julio de 2014, proporcionando esta fecha como entrada al `energy` en el patrón `[desde fecha]: [hasta fecha]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julio](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/1-Introduction/images/july-2014.png)

    ¡Un gráfico hermoso! Observa estos gráficos y ve si puedes determinar alguna de las características mencionadas anteriormente. ¿Qué podemos deducir al visualizar los datos?

En la próxima lección, crearás un modelo ARIMA para generar algunas predicciones.

---

## 🚀Desafío

Haz una lista de todas las industrias y áreas de investigación que se te ocurran que podrían beneficiarse de la predicción de series temporales. ¿Puedes pensar en una aplicación de estas técnicas en las artes? ¿En econometría? ¿Ecología? ¿Retail? ¿Industria? ¿Finanzas? ¿Dónde más?

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Aunque no los cubriremos aquí, las redes neuronales a veces se utilizan para mejorar los métodos clásicos de predicción de series temporales. Lee más sobre ellas [en este artículo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412).

## Tarea

[Visualiza más series temporales](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T22:14:48+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "es"
}
-->
# Pronóstico de series temporales con ARIMA

En la lección anterior, aprendiste un poco sobre el pronóstico de series temporales y cargaste un conjunto de datos que muestra las fluctuaciones de la carga eléctrica a lo largo de un período de tiempo.

[![Introducción a ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introducción a ARIMA")

> 🎥 Haz clic en la imagen de arriba para ver un video: Una breve introducción a los modelos ARIMA. El ejemplo está hecho en R, pero los conceptos son universales.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

En esta lección, descubrirás una forma específica de construir modelos con [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Los modelos ARIMA son particularmente adecuados para ajustar datos que muestran [no estacionariedad](https://wikipedia.org/wiki/Stationary_process).

## Conceptos generales

Para trabajar con ARIMA, hay algunos conceptos que necesitas conocer:

- 🎓 **Estacionariedad**. En un contexto estadístico, la estacionariedad se refiere a datos cuya distribución no cambia al desplazarse en el tiempo. Los datos no estacionarios, por lo tanto, muestran fluctuaciones debido a tendencias que deben transformarse para ser analizadas. La estacionalidad, por ejemplo, puede introducir fluctuaciones en los datos y puede eliminarse mediante un proceso de 'diferenciación estacional'.

- 🎓 **[Diferenciación](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. La diferenciación de datos, nuevamente desde un contexto estadístico, se refiere al proceso de transformar datos no estacionarios para hacerlos estacionarios eliminando su tendencia no constante. "La diferenciación elimina los cambios en el nivel de una serie temporal, eliminando la tendencia y la estacionalidad y, en consecuencia, estabilizando la media de la serie temporal." [Artículo de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA en el contexto de series temporales

Desglosemos las partes de ARIMA para entender mejor cómo nos ayuda a modelar series temporales y hacer predicciones.

- **AR - de AutoRegresivo**. Los modelos autorregresivos, como su nombre lo indica, miran 'hacia atrás' en el tiempo para analizar valores previos en tus datos y hacer suposiciones sobre ellos. Estos valores previos se llaman 'lags' (rezagos). Un ejemplo sería un conjunto de datos que muestra las ventas mensuales de lápices. El total de ventas de cada mes se consideraría una 'variable evolutiva' en el conjunto de datos. Este modelo se construye como "la variable evolutiva de interés se regresa sobre sus propios valores rezagados (es decir, valores anteriores)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - de Integrado**. A diferencia de los modelos similares 'ARMA', la 'I' en ARIMA se refiere a su aspecto *[integrado](https://wikipedia.org/wiki/Order_of_integration)*. Los datos se 'integran' cuando se aplican pasos de diferenciación para eliminar la no estacionariedad.

- **MA - de Media Móvil**. El aspecto de [media móvil](https://wikipedia.org/wiki/Moving-average_model) de este modelo se refiere a la variable de salida que se determina observando los valores actuales y pasados de los rezagos.

En resumen: ARIMA se utiliza para ajustar un modelo lo más cerca posible a la forma especial de los datos de series temporales.

## Ejercicio - construir un modelo ARIMA

Abre la carpeta [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) en esta lección y encuentra el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Ejecuta el notebook para cargar la biblioteca de Python `statsmodels`; la necesitarás para los modelos ARIMA.

1. Carga las bibliotecas necesarias.

1. Ahora, carga varias bibliotecas más útiles para graficar datos:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. Carga los datos del archivo `/data/energy.csv` en un dataframe de Pandas y échales un vistazo:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Grafica todos los datos de energía disponibles desde enero de 2012 hasta diciembre de 2014. No debería haber sorpresas, ya que vimos estos datos en la lección anterior:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ¡Ahora, construyamos un modelo!

### Crear conjuntos de datos de entrenamiento y prueba

Ahora que tus datos están cargados, puedes separarlos en conjuntos de entrenamiento y prueba. Entrenarás tu modelo en el conjunto de entrenamiento. Como de costumbre, después de que el modelo haya terminado de entrenar, evaluarás su precisión utilizando el conjunto de prueba. Debes asegurarte de que el conjunto de prueba cubra un período posterior en el tiempo al conjunto de entrenamiento para garantizar que el modelo no obtenga información de períodos futuros.

1. Asigna un período de dos meses desde el 1 de septiembre hasta el 31 de octubre de 2014 al conjunto de entrenamiento. El conjunto de prueba incluirá el período de dos meses del 1 de noviembre al 31 de diciembre de 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Dado que estos datos reflejan el consumo diario de energía, hay un fuerte patrón estacional, pero el consumo es más similar al consumo de días más recientes.

1. Visualiza las diferencias:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![datos de entrenamiento y prueba](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/images/train-test.png)

    Por lo tanto, usar una ventana de tiempo relativamente pequeña para entrenar los datos debería ser suficiente.

    > Nota: Dado que la función que usamos para ajustar el modelo ARIMA utiliza validación dentro de la muestra durante el ajuste, omitiremos los datos de validación.

### Preparar los datos para el entrenamiento

Ahora necesitas preparar los datos para el entrenamiento realizando un filtrado y escalado de tus datos. Filtra tu conjunto de datos para incluir solo los períodos de tiempo y columnas que necesitas, y escala los datos para asegurarte de que estén proyectados en el intervalo 0,1.

1. Filtra el conjunto de datos original para incluir solo los períodos de tiempo mencionados por conjunto e incluyendo únicamente la columna necesaria 'load' más la fecha:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Puedes ver la forma de los datos:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Escala los datos para que estén en el rango (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualiza los datos originales vs. los escalados:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/images/original.png)

    > Los datos originales

    ![escalados](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/images/scaled.png)

    > Los datos escalados

1. Ahora que has calibrado los datos escalados, puedes escalar los datos de prueba:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementar ARIMA

¡Es hora de implementar ARIMA! Ahora usarás la biblioteca `statsmodels` que instalaste anteriormente.

Ahora necesitas seguir varios pasos:

   1. Define el modelo llamando a `SARIMAX()` y pasando los parámetros del modelo: parámetros p, d y q, y parámetros P, D y Q.
   2. Prepara el modelo para los datos de entrenamiento llamando a la función `fit()`.
   3. Realiza predicciones llamando a la función `forecast()` y especificando el número de pasos (el `horizonte`) a pronosticar.

> 🎓 ¿Para qué son todos estos parámetros? En un modelo ARIMA hay 3 parámetros que se utilizan para ayudar a modelar los aspectos principales de una serie temporal: estacionalidad, tendencia y ruido. Estos parámetros son:

`p`: el parámetro asociado con el aspecto autorregresivo del modelo, que incorpora valores *pasados*.  
`d`: el parámetro asociado con la parte integrada del modelo, que afecta la cantidad de *diferenciación* (🎓 ¿recuerdas la diferenciación 👆?) que se aplica a una serie temporal.  
`q`: el parámetro asociado con la parte de media móvil del modelo.  

> Nota: Si tus datos tienen un aspecto estacional - como en este caso -, usamos un modelo ARIMA estacional (SARIMA). En ese caso, necesitas usar otro conjunto de parámetros: `P`, `D` y `Q`, que describen las mismas asociaciones que `p`, `d` y `q`, pero corresponden a los componentes estacionales del modelo.

1. Comienza configurando tu valor de horizonte preferido. Probemos con 3 horas:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Seleccionar los mejores valores para los parámetros de un modelo ARIMA puede ser un desafío, ya que es algo subjetivo y requiere tiempo. Podrías considerar usar una función `auto_arima()` de la [biblioteca `pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Por ahora, intenta algunas selecciones manuales para encontrar un buen modelo.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Se imprime una tabla de resultados.

¡Has construido tu primer modelo! Ahora necesitamos encontrar una forma de evaluarlo.

### Evaluar tu modelo

Para evaluar tu modelo, puedes realizar la llamada validación `walk forward`. En la práctica, los modelos de series temporales se reentrenan cada vez que se dispone de nuevos datos. Esto permite que el modelo haga el mejor pronóstico en cada paso de tiempo.

Comenzando al principio de la serie temporal con esta técnica, entrena el modelo en el conjunto de datos de entrenamiento. Luego realiza una predicción en el siguiente paso de tiempo. La predicción se evalúa en comparación con el valor conocido. El conjunto de entrenamiento se amplía para incluir el valor conocido y el proceso se repite.

> Nota: Deberías mantener fija la ventana del conjunto de entrenamiento para un entrenamiento más eficiente, de modo que cada vez que agregues una nueva observación al conjunto de entrenamiento, elimines la observación del principio del conjunto.

Este proceso proporciona una estimación más robusta de cómo se desempeñará el modelo en la práctica. Sin embargo, tiene el costo computacional de crear tantos modelos. Esto es aceptable si los datos son pequeños o si el modelo es simple, pero podría ser un problema a gran escala.

La validación walk-forward es el estándar de oro para la evaluación de modelos de series temporales y se recomienda para tus propios proyectos.

1. Primero, crea un punto de datos de prueba para cada paso del HORIZON.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Los datos se desplazan horizontalmente según su punto de horizonte.

1. Realiza predicciones en tus datos de prueba utilizando este enfoque de ventana deslizante en un bucle del tamaño de la longitud de los datos de prueba:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Puedes observar el entrenamiento en curso:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Compara las predicciones con la carga real:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Salida  
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Observa la predicción de los datos horarios en comparación con la carga real. ¿Qué tan precisa es?

### Verificar la precisión del modelo

Verifica la precisión de tu modelo probando su error porcentual absoluto medio (MAPE) en todas las predicciones.
> **🧮 Muéstrame las matemáticas**
>
> ![MAPE](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) se utiliza para mostrar la precisión de las predicciones como una proporción definida por la fórmula anterior. La diferencia entre el valor real y el valor predicho se divide por el valor real.  
> "El valor absoluto en este cálculo se suma para cada punto pronosticado en el tiempo y se divide por el número de puntos ajustados n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Expresar la ecuación en código:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Calcular el MAPE de un paso:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE de pronóstico de un paso:  0.5570581332313952 %

1. Imprimir el MAPE del pronóstico de múltiples pasos:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Un número bajo es lo mejor: considera que un pronóstico con un MAPE de 10 está desviado en un 10%.

1. Pero como siempre, es más fácil ver este tipo de medición de precisión de forma visual, así que vamos a graficarlo:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![un modelo de series temporales](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Un gráfico muy bueno, mostrando un modelo con buena precisión. ¡Bien hecho!

---

## 🚀Desafío

Investiga las formas de probar la precisión de un modelo de series temporales. En esta lección hablamos sobre el MAPE, pero ¿hay otros métodos que podrías usar? Investígalos y anótalos. Un documento útil se puede encontrar [aquí](https://otexts.com/fpp2/accuracy.html)

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Esta lección solo toca los conceptos básicos de la predicción de series temporales con ARIMA. Tómate un tiempo para profundizar en tu conocimiento explorando [este repositorio](https://microsoft.github.io/forecasting/) y sus diversos tipos de modelos para aprender otras formas de construir modelos de series temporales.

## Tarea

[Un nuevo modelo ARIMA](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T22:16:07+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "es"
}
-->
# Pronóstico de Series Temporales con Support Vector Regressor

En la lección anterior, aprendiste a usar el modelo ARIMA para realizar predicciones de series temporales. Ahora explorarás el modelo Support Vector Regressor, que es un modelo de regresión utilizado para predecir datos continuos.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/) 

## Introducción

En esta lección, descubrirás una forma específica de construir modelos con [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) para regresión, o **SVR: Support Vector Regressor**. 

### SVR en el contexto de series temporales [^1]

Antes de entender la importancia de SVR en la predicción de series temporales, aquí tienes algunos conceptos importantes que necesitas conocer:

- **Regresión:** Técnica de aprendizaje supervisado para predecir valores continuos a partir de un conjunto de entradas. La idea es ajustar una curva (o línea) en el espacio de características que tenga el mayor número de puntos de datos. [Haz clic aquí](https://en.wikipedia.org/wiki/Regression_analysis) para más información.
- **Support Vector Machine (SVM):** Un tipo de modelo de aprendizaje supervisado utilizado para clasificación, regresión y detección de valores atípicos. El modelo es un hiperplano en el espacio de características, que en el caso de clasificación actúa como un límite, y en el caso de regresión actúa como la línea de mejor ajuste. En SVM, generalmente se utiliza una función Kernel para transformar el conjunto de datos a un espacio de mayor número de dimensiones, de modo que puedan ser fácilmente separables. [Haz clic aquí](https://en.wikipedia.org/wiki/Support-vector_machine) para más información sobre SVMs.
- **Support Vector Regressor (SVR):** Un tipo de SVM, que encuentra la línea de mejor ajuste (que en el caso de SVM es un hiperplano) que tiene el mayor número de puntos de datos.

### ¿Por qué SVR? [^1]

En la última lección aprendiste sobre ARIMA, que es un método estadístico lineal muy exitoso para pronosticar datos de series temporales. Sin embargo, en muchos casos, los datos de series temporales tienen *no linealidad*, que no puede ser modelada por modelos lineales. En tales casos, la capacidad de SVM para considerar la no linealidad en los datos para tareas de regresión hace que SVR sea exitoso en el pronóstico de series temporales.

## Ejercicio - construir un modelo SVR

Los primeros pasos para la preparación de datos son los mismos que en la lección anterior sobre [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Abre la carpeta [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) en esta lección y encuentra el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Ejecuta el notebook e importa las bibliotecas necesarias: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Carga los datos del archivo `/data/energy.csv` en un dataframe de Pandas y échales un vistazo: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Grafica todos los datos de energía disponibles desde enero de 2012 hasta diciembre de 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![datos completos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/3-SVR/images/full-data.png)

   Ahora, construyamos nuestro modelo SVR.

### Crear conjuntos de entrenamiento y prueba

Ahora que tus datos están cargados, puedes separarlos en conjuntos de entrenamiento y prueba. Luego, reformatearás los datos para crear un conjunto de datos basado en pasos de tiempo, que será necesario para el SVR. Entrenarás tu modelo en el conjunto de entrenamiento. Una vez que el modelo haya terminado de entrenarse, evaluarás su precisión en el conjunto de entrenamiento, el conjunto de prueba y luego en el conjunto de datos completo para ver el rendimiento general. Debes asegurarte de que el conjunto de prueba cubra un período posterior en el tiempo al conjunto de entrenamiento para garantizar que el modelo no obtenga información de períodos futuros [^2] (una situación conocida como *sobreajuste*).

1. Asigna un período de dos meses, del 1 de septiembre al 31 de octubre de 2014, al conjunto de entrenamiento. El conjunto de prueba incluirá el período de dos meses del 1 de noviembre al 31 de diciembre de 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiza las diferencias: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![datos de entrenamiento y prueba](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/3-SVR/images/train-test.png)

### Preparar los datos para el entrenamiento

Ahora necesitas preparar los datos para el entrenamiento realizando un filtrado y escalado de tus datos. Filtra tu conjunto de datos para incluir solo los períodos de tiempo y columnas necesarios, y escala los datos para asegurarte de que estén proyectados en el intervalo 0,1.

1. Filtra el conjunto de datos original para incluir solo los períodos de tiempo mencionados anteriormente por conjunto e incluyendo únicamente la columna 'load' más la fecha: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. Escala los datos de entrenamiento para que estén en el rango (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Ahora, escala los datos de prueba: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Crear datos con pasos de tiempo [^1]

Para el SVR, transformas los datos de entrada para que tengan la forma `[batch, timesteps]`. Por lo tanto, reformateas los `train_data` y `test_data` existentes de manera que haya una nueva dimensión que se refiera a los pasos de tiempo. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Para este ejemplo, tomamos `timesteps = 5`. Por lo tanto, las entradas al modelo son los datos de los primeros 4 pasos de tiempo, y la salida serán los datos del quinto paso de tiempo.

```python
timesteps=5
```

Convirtiendo los datos de entrenamiento a un tensor 2D usando comprensión de listas anidadas:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Convirtiendo los datos de prueba a un tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Seleccionando entradas y salidas de los datos de entrenamiento y prueba:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Implementar SVR [^1]

Ahora es momento de implementar SVR. Para leer más sobre esta implementación, puedes consultar [esta documentación](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Para nuestra implementación, seguimos estos pasos:

  1. Define el modelo llamando a `SVR()` y pasando los hiperparámetros del modelo: kernel, gamma, c y epsilon.
  2. Prepara el modelo para los datos de entrenamiento llamando a la función `fit()`.
  3. Realiza predicciones llamando a la función `predict()`.

Ahora creamos un modelo SVR. Aquí usamos el [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), y configuramos los hiperparámetros gamma, C y epsilon como 0.5, 10 y 0.05 respectivamente.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Ajustar el modelo a los datos de entrenamiento [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Realizar predicciones con el modelo [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

¡Has construido tu SVR! Ahora necesitamos evaluarlo.

### Evaluar tu modelo [^1]

Para la evaluación, primero escalaremos los datos de vuelta a nuestra escala original. Luego, para verificar el rendimiento, graficaremos la serie temporal original y la predicha, y también imprimiremos el resultado de MAPE.

Escala la salida predicha y la original:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Verificar el rendimiento del modelo en los datos de entrenamiento y prueba [^1]

Extraemos las marcas de tiempo del conjunto de datos para mostrarlas en el eje x de nuestro gráfico. Ten en cuenta que estamos utilizando los primeros ```timesteps-1``` valores como entrada para la primera salida, por lo que las marcas de tiempo para la salida comenzarán después de eso.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Grafica las predicciones para los datos de entrenamiento:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![predicción de datos de entrenamiento](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/3-SVR/images/train-data-predict.png)

Imprime el MAPE para los datos de entrenamiento:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Grafica las predicciones para los datos de prueba:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![predicción de datos de prueba](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/3-SVR/images/test-data-predict.png)

Imprime el MAPE para los datos de prueba:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 ¡Tienes un muy buen resultado en el conjunto de datos de prueba!

### Verificar el rendimiento del modelo en el conjunto de datos completo [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![predicción de datos completos](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Muy buenos gráficos, mostrando un modelo con buena precisión. ¡Bien hecho!

---

## 🚀Desafío

- Intenta ajustar los hiperparámetros (gamma, C, epsilon) al crear el modelo y evalúalo en los datos para ver qué conjunto de hiperparámetros da los mejores resultados en los datos de prueba. Para saber más sobre estos hiperparámetros, puedes consultar el documento [aquí](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Intenta usar diferentes funciones kernel para el modelo y analiza su rendimiento en el conjunto de datos. Un documento útil se encuentra [aquí](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Intenta usar diferentes valores para `timesteps` para que el modelo mire hacia atrás y haga predicciones.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

Esta lección fue para introducir la aplicación de SVR para el pronóstico de series temporales. Para leer más sobre SVR, puedes consultar [este blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Esta [documentación en scikit-learn](https://scikit-learn.org/stable/modules/svm.html) proporciona una explicación más completa sobre SVMs en general, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) y también otros detalles de implementación como las diferentes [funciones kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) que se pueden usar y sus parámetros.

## Tarea

[Un nuevo modelo SVR](assignment.md)

## Créditos

[^1]: El texto, código y salida en esta sección fueron contribuidos por [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: El texto, código y salida en esta sección fueron tomados de [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---


# Aprendizaje por Refuerzo

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-04T00:14:18+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "es"
}
-->
# Introducción al aprendizaje por refuerzo

El aprendizaje por refuerzo, RL, se considera uno de los paradigmas básicos del aprendizaje automático, junto con el aprendizaje supervisado y el aprendizaje no supervisado. RL trata sobre decisiones: tomar las decisiones correctas o, al menos, aprender de ellas.

Imagina que tienes un entorno simulado como el mercado de valores. ¿Qué sucede si impones una regulación específica? ¿Tiene un efecto positivo o negativo? Si ocurre algo negativo, necesitas tomar este _refuerzo negativo_, aprender de ello y cambiar de rumbo. Si el resultado es positivo, necesitas construir sobre ese _refuerzo positivo_.

![Pedro y el lobo](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/peter.779730f9ba3a8a8d.webp)

> ¡Pedro y sus amigos necesitan escapar del lobo hambriento! Imagen por [Jen Looper](https://twitter.com/jenlooper)

## Tema regional: Pedro y el Lobo (Rusia)

[Pedro y el Lobo](https://es.wikipedia.org/wiki/Pedro_y_el_lobo) es un cuento musical escrito por el compositor ruso [Sergei Prokofiev](https://es.wikipedia.org/wiki/Sergu%C3%A9i_Prok%C3%B3fiev). Es una historia sobre el joven pionero Pedro, quien valientemente sale de su casa hacia el claro del bosque para perseguir al lobo. En esta sección, entrenaremos algoritmos de aprendizaje automático que ayudarán a Pedro:

- **Explorar** el área circundante y construir un mapa de navegación óptimo.
- **Aprender** a usar un monopatín y mantener el equilibrio en él, para moverse más rápido.

[![Pedro y el Lobo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Haz clic en la imagen de arriba para escuchar Pedro y el Lobo de Prokofiev

## Aprendizaje por refuerzo

En secciones anteriores, has visto dos ejemplos de problemas de aprendizaje automático:

- **Supervisado**, donde tenemos conjuntos de datos que sugieren soluciones de muestra para el problema que queremos resolver. [Clasificación](../4-Classification/README.md) y [regresión](../2-Regression/README.md) son tareas de aprendizaje supervisado.
- **No supervisado**, en el que no tenemos datos de entrenamiento etiquetados. El principal ejemplo de aprendizaje no supervisado es [Agrupamiento](../5-Clustering/README.md).

En esta sección, te presentaremos un nuevo tipo de problema de aprendizaje que no requiere datos de entrenamiento etiquetados. Hay varios tipos de problemas de este tipo:

- **[Aprendizaje semisupervisado](https://es.wikipedia.org/wiki/Aprendizaje_semisupervisado)**, donde tenemos una gran cantidad de datos no etiquetados que pueden usarse para preentrenar el modelo.
- **[Aprendizaje por refuerzo](https://es.wikipedia.org/wiki/Aprendizaje_por_refuerzo)**, en el que un agente aprende cómo comportarse realizando experimentos en algún entorno simulado.

### Ejemplo - videojuego

Supongamos que quieres enseñar a una computadora a jugar un juego, como ajedrez o [Super Mario](https://es.wikipedia.org/wiki/Super_Mario). Para que la computadora juegue, necesitamos que prediga qué movimiento realizar en cada estado del juego. Aunque esto pueda parecer un problema de clasificación, no lo es, porque no tenemos un conjunto de datos con estados y acciones correspondientes. Aunque podríamos tener algunos datos como partidas de ajedrez existentes o grabaciones de jugadores jugando Super Mario, es probable que esos datos no cubran suficientemente una gran cantidad de estados posibles.

En lugar de buscar datos existentes del juego, el **Aprendizaje por Refuerzo** (RL) se basa en la idea de *hacer que la computadora juegue* muchas veces y observar el resultado. Por lo tanto, para aplicar el Aprendizaje por Refuerzo, necesitamos dos cosas:

- **Un entorno** y **un simulador** que nos permitan jugar muchas veces. Este simulador definiría todas las reglas del juego, así como los posibles estados y acciones.

- **Una función de recompensa**, que nos indique qué tan bien lo hicimos durante cada movimiento o partida.

La principal diferencia entre otros tipos de aprendizaje automático y RL es que en RL típicamente no sabemos si ganamos o perdemos hasta que terminamos el juego. Por lo tanto, no podemos decir si un movimiento en particular es bueno o no: solo recibimos una recompensa al final del juego. Y nuestro objetivo es diseñar algoritmos que nos permitan entrenar un modelo bajo condiciones inciertas. Aprenderemos sobre un algoritmo de RL llamado **Q-learning**.

## Lecciones

1. [Introducción al aprendizaje por refuerzo y Q-Learning](1-QLearning/README.md)
2. [Uso de un entorno de simulación gym](2-Gym/README.md)

## Créditos

"Introducción al Aprendizaje por Refuerzo" fue escrito con ♥️ por [Dmitry Soshnikov](http://soshnikov.com)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-04T22:25:29+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "es"
}
-->
# Introducción al Aprendizaje por Refuerzo y Q-Learning

![Resumen del refuerzo en el aprendizaje automático en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-reinforcement.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

El aprendizaje por refuerzo implica tres conceptos importantes: el agente, algunos estados y un conjunto de acciones por estado. Al ejecutar una acción en un estado específico, el agente recibe una recompensa. Imagina nuevamente el videojuego Super Mario. Tú eres Mario, estás en un nivel del juego, parado junto al borde de un acantilado. Sobre ti hay una moneda. Tú, siendo Mario, en un nivel del juego, en una posición específica... ese es tu estado. Moverte un paso hacia la derecha (una acción) te llevará al borde y te dará una puntuación numérica baja. Sin embargo, presionar el botón de salto te permitirá ganar un punto y seguir vivo. Ese es un resultado positivo y debería otorgarte una puntuación numérica positiva.

Usando aprendizaje por refuerzo y un simulador (el juego), puedes aprender a jugar para maximizar la recompensa, que es permanecer vivo y obtener la mayor cantidad de puntos posible.

[![Introducción al Aprendizaje por Refuerzo](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Haz clic en la imagen de arriba para escuchar a Dmitry hablar sobre el Aprendizaje por Refuerzo

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Prerrequisitos y Configuración

En esta lección, experimentaremos con algo de código en Python. Deberías poder ejecutar el código del Jupyter Notebook de esta lección, ya sea en tu computadora o en la nube.

Puedes abrir [el notebook de la lección](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) y seguir esta lección para construir.

> **Nota:** Si estás abriendo este código desde la nube, también necesitas obtener el archivo [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que se utiliza en el código del notebook. Agrégalo al mismo directorio que el notebook.

## Introducción

En esta lección, exploraremos el mundo de **[Pedro y el Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirado en un cuento musical de hadas de un compositor ruso, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Usaremos **Aprendizaje por Refuerzo** para permitir que Pedro explore su entorno, recoja manzanas deliciosas y evite encontrarse con el lobo.

El **Aprendizaje por Refuerzo** (RL) es una técnica de aprendizaje que nos permite aprender un comportamiento óptimo de un **agente** en algún **entorno** realizando muchos experimentos. Un agente en este entorno debe tener algún **objetivo**, definido por una **función de recompensa**.

## El entorno

Para simplificar, consideremos el mundo de Pedro como un tablero cuadrado de tamaño `ancho` x `alto`, como este:

![Entorno de Pedro](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/1-QLearning/images/environment.png)

Cada celda en este tablero puede ser:

* **suelo**, sobre el cual Pedro y otras criaturas pueden caminar.
* **agua**, sobre la cual obviamente no puedes caminar.
* un **árbol** o **hierba**, un lugar donde puedes descansar.
* una **manzana**, que representa algo que Pedro estaría encantado de encontrar para alimentarse.
* un **lobo**, que es peligroso y debe evitarse.

Hay un módulo de Python separado, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que contiene el código para trabajar con este entorno. Dado que este código no es importante para entender nuestros conceptos, importaremos el módulo y lo usaremos para crear el tablero de muestra (bloque de código 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Este código debería imprimir una imagen del entorno similar a la anterior.

## Acciones y política

En nuestro ejemplo, el objetivo de Pedro sería encontrar una manzana, mientras evita al lobo y otros obstáculos. Para lograr esto, esencialmente puede caminar por el tablero hasta encontrar una manzana.

Por lo tanto, en cualquier posición, puede elegir entre una de las siguientes acciones: arriba, abajo, izquierda y derecha.

Definiremos esas acciones como un diccionario y las mapearemos a pares de cambios de coordenadas correspondientes. Por ejemplo, moverse a la derecha (`R`) correspondería a un par `(1,0)`. (bloque de código 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

En resumen, la estrategia y el objetivo de este escenario son los siguientes:

- **La estrategia** de nuestro agente (Pedro) está definida por una llamada **política**. Una política es una función que devuelve la acción en cualquier estado dado. En nuestro caso, el estado del problema está representado por el tablero, incluida la posición actual del jugador.

- **El objetivo** del aprendizaje por refuerzo es eventualmente aprender una buena política que nos permita resolver el problema de manera eficiente. Sin embargo, como línea base, consideremos la política más simple llamada **camino aleatorio**.

## Camino aleatorio

Primero resolvamos nuestro problema implementando una estrategia de camino aleatorio. Con el camino aleatorio, elegiremos aleatoriamente la siguiente acción de las acciones permitidas, hasta que lleguemos a la manzana (bloque de código 3).

1. Implementa el camino aleatorio con el siguiente código:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    La llamada a `walk` debería devolver la longitud del camino correspondiente, que puede variar de una ejecución a otra.

1. Ejecuta el experimento de camino varias veces (digamos, 100) y muestra las estadísticas resultantes (bloque de código 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Nota que la longitud promedio de un camino es de alrededor de 30-40 pasos, lo cual es bastante, dado que la distancia promedio a la manzana más cercana es de alrededor de 5-6 pasos.

    También puedes ver cómo se mueve Pedro durante el camino aleatorio:

    ![Camino aleatorio de Pedro](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/1-QLearning/images/random_walk.gif)

## Función de recompensa

Para hacer nuestra política más inteligente, necesitamos entender qué movimientos son "mejores" que otros. Para hacer esto, necesitamos definir nuestro objetivo.

El objetivo puede definirse en términos de una **función de recompensa**, que devolverá algún valor de puntuación para cada estado. Cuanto mayor sea el número, mejor será la función de recompensa. (bloque de código 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Lo interesante de las funciones de recompensa es que en la mayoría de los casos, *solo se nos da una recompensa sustancial al final del juego*. Esto significa que nuestro algoritmo debería recordar los pasos "buenos" que conducen a una recompensa positiva al final y aumentar su importancia. De manera similar, todos los movimientos que conducen a malos resultados deberían desalentarse.

## Q-Learning

El algoritmo que discutiremos aquí se llama **Q-Learning**. En este algoritmo, la política está definida por una función (o una estructura de datos) llamada **Q-Table**. Registra la "bondad" de cada una de las acciones en un estado dado.

Se llama Q-Table porque a menudo es conveniente representarla como una tabla o un arreglo multidimensional. Dado que nuestro tablero tiene dimensiones `ancho` x `alto`, podemos representar la Q-Table usando un arreglo numpy con forma `ancho` x `alto` x `len(actions)`: (bloque de código 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Nota que inicializamos todos los valores de la Q-Table con un valor igual, en nuestro caso - 0.25. Esto corresponde a la política de "camino aleatorio", porque todos los movimientos en cada estado son igualmente buenos. Podemos pasar la Q-Table a la función `plot` para visualizar la tabla en el tablero: `m.plot(Q)`.

![Entorno de Pedro](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/1-QLearning/images/env_init.png)

En el centro de cada celda hay una "flecha" que indica la dirección preferida de movimiento. Dado que todas las direcciones son iguales, se muestra un punto.

Ahora necesitamos ejecutar la simulación, explorar nuestro entorno y aprender una mejor distribución de valores de la Q-Table, lo que nos permitirá encontrar el camino hacia la manzana mucho más rápido.

## Esencia del Q-Learning: Ecuación de Bellman

Una vez que comenzamos a movernos, cada acción tendrá una recompensa correspondiente, es decir, teóricamente podemos seleccionar la siguiente acción basada en la recompensa inmediata más alta. Sin embargo, en la mayoría de los estados, el movimiento no logrará nuestro objetivo de alcanzar la manzana, y por lo tanto no podemos decidir inmediatamente qué dirección es mejor.

> Recuerda que no importa el resultado inmediato, sino el resultado final, que obtendremos al final de la simulación.

Para tener en cuenta esta recompensa diferida, necesitamos usar los principios de la **[programación dinámica](https://en.wikipedia.org/wiki/Dynamic_programming)**, que nos permiten pensar en nuestro problema de manera recursiva.

Supongamos que ahora estamos en el estado *s*, y queremos movernos al siguiente estado *s'*. Al hacerlo, recibiremos la recompensa inmediata *r(s,a)*, definida por la función de recompensa, más alguna recompensa futura. Si suponemos que nuestra Q-Table refleja correctamente la "atractividad" de cada acción, entonces en el estado *s'* elegiremos una acción *a'* que corresponda al valor máximo de *Q(s',a')*. Así, la mejor recompensa futura posible que podríamos obtener en el estado *s* se definirá como `max`

## Comprobando la política

Dado que la Q-Table enumera la "atractividad" de cada acción en cada estado, es bastante fácil usarla para definir la navegación eficiente en nuestro mundo. En el caso más simple, podemos seleccionar la acción correspondiente al valor más alto de la Q-Table: (bloque de código 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Si pruebas el código anterior varias veces, puede que notes que a veces se "queda colgado" y necesitas presionar el botón STOP en el notebook para interrumpirlo. Esto ocurre porque podría haber situaciones en las que dos estados "apuntan" el uno al otro en términos de valor óptimo de Q-Value, en cuyo caso el agente termina moviéndose entre esos estados indefinidamente.

## 🚀Desafío

> **Tarea 1:** Modifica la función `walk` para limitar la longitud máxima del camino a un cierto número de pasos (por ejemplo, 100), y observa cómo el código anterior devuelve este valor de vez en cuando.

> **Tarea 2:** Modifica la función `walk` para que no regrese a los lugares donde ya ha estado previamente. Esto evitará que `walk` entre en bucles, sin embargo, el agente aún puede terminar "atrapado" en una ubicación de la que no puede escapar.

## Navegación

Una política de navegación mejor sería la que usamos durante el entrenamiento, que combina explotación y exploración. En esta política, seleccionaremos cada acción con cierta probabilidad, proporcional a los valores en la Q-Table. Esta estrategia aún puede hacer que el agente regrese a una posición que ya ha explorado, pero, como puedes ver en el código a continuación, resulta en un camino promedio muy corto hacia la ubicación deseada (recuerda que `print_statistics` ejecuta la simulación 100 veces): (bloque de código 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Después de ejecutar este código, deberías obtener una longitud promedio de camino mucho más pequeña que antes, en el rango de 3-6.

## Investigando el proceso de aprendizaje

Como hemos mencionado, el proceso de aprendizaje es un equilibrio entre la exploración y la explotación del conocimiento adquirido sobre la estructura del espacio del problema. Hemos visto que los resultados del aprendizaje (la capacidad de ayudar a un agente a encontrar un camino corto hacia el objetivo) han mejorado, pero también es interesante observar cómo se comporta la longitud promedio del camino durante el proceso de aprendizaje:

## Los aprendizajes pueden resumirse como:

- **La longitud promedio del camino aumenta**. Lo que vemos aquí es que al principio, la longitud promedio del camino aumenta. Esto probablemente se debe al hecho de que cuando no sabemos nada sobre el entorno, es probable que quedemos atrapados en estados desfavorables, como agua o lobos. A medida que aprendemos más y comenzamos a usar este conocimiento, podemos explorar el entorno por más tiempo, pero aún no sabemos muy bien dónde están las manzanas.

- **La longitud del camino disminuye a medida que aprendemos más**. Una vez que aprendemos lo suficiente, se vuelve más fácil para el agente alcanzar el objetivo, y la longitud del camino comienza a disminuir. Sin embargo, aún estamos abiertos a la exploración, por lo que a menudo nos desviamos del mejor camino y exploramos nuevas opciones, haciendo que el camino sea más largo de lo óptimo.

- **La longitud aumenta abruptamente**. Lo que también observamos en este gráfico es que en algún momento, la longitud aumentó abruptamente. Esto indica la naturaleza estocástica del proceso, y que en algún momento podemos "estropear" los coeficientes de la Q-Table sobrescribiéndolos con nuevos valores. Esto idealmente debería minimizarse disminuyendo la tasa de aprendizaje (por ejemplo, hacia el final del entrenamiento, solo ajustamos los valores de la Q-Table por un pequeño valor).

En general, es importante recordar que el éxito y la calidad del proceso de aprendizaje dependen significativamente de parámetros como la tasa de aprendizaje, la disminución de la tasa de aprendizaje y el factor de descuento. A menudo se les llama **hiperparámetros**, para distinguirlos de los **parámetros**, que optimizamos durante el entrenamiento (por ejemplo, los coeficientes de la Q-Table). El proceso de encontrar los mejores valores de hiperparámetros se llama **optimización de hiperparámetros**, y merece un tema aparte.

## [Cuestionario post-clase](https://ff-quizzes.netlify.app/en/ml/)

## Tarea 
[Un Mundo Más Realista](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-04T22:26:14+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "es"
}
-->
# CartPole Patinaje

El problema que resolvimos en la lección anterior puede parecer un problema de juguete, sin mucha aplicación en escenarios de la vida real. Sin embargo, este no es el caso, ya que muchos problemas del mundo real comparten características similares, como jugar al ajedrez o al Go. Son similares porque también tenemos un tablero con reglas definidas y un **estado discreto**.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

En esta lección aplicaremos los mismos principios de Q-Learning a un problema con un **estado continuo**, es decir, un estado definido por uno o más números reales. Abordaremos el siguiente problema:

> **Problema**: Si Pedro quiere escapar del lobo, necesita aprender a moverse más rápido. Veremos cómo Pedro puede aprender a patinar, en particular, a mantener el equilibrio, utilizando Q-Learning.

![¡La gran escapada!](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/images/escape.png)

> ¡Pedro y sus amigos se ponen creativos para escapar del lobo! Imagen de [Jen Looper](https://twitter.com/jenlooper)

Usaremos una versión simplificada del equilibrio conocida como el problema del **CartPole**. En el mundo del CartPole, tenemos un deslizador horizontal que puede moverse a la izquierda o a la derecha, y el objetivo es equilibrar un palo vertical sobre el deslizador.

## Prerrequisitos

En esta lección, utilizaremos una biblioteca llamada **OpenAI Gym** para simular diferentes **entornos**. Puedes ejecutar el código de esta lección localmente (por ejemplo, desde Visual Studio Code), en cuyo caso la simulación se abrirá en una nueva ventana. Si ejecutas el código en línea, es posible que necesites hacer algunos ajustes, como se describe [aquí](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

En la lección anterior, las reglas del juego y el estado estaban definidos por la clase `Board` que creamos nosotros mismos. Aquí utilizaremos un **entorno de simulación** especial, que simulará la física detrás del palo en equilibrio. Uno de los entornos de simulación más populares para entrenar algoritmos de aprendizaje por refuerzo se llama [Gym](https://gym.openai.com/), mantenido por [OpenAI](https://openai.com/). Usando este Gym, podemos crear diferentes **entornos**, desde una simulación de CartPole hasta juegos de Atari.

> **Nota**: Puedes ver otros entornos disponibles en OpenAI Gym [aquí](https://gym.openai.com/envs/#classic_control).

Primero, instalemos Gym e importemos las bibliotecas necesarias (bloque de código 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ejercicio - inicializar un entorno de CartPole

Para trabajar con el problema de equilibrio de CartPole, necesitamos inicializar el entorno correspondiente. Cada entorno está asociado con:

- Un **espacio de observación** que define la estructura de la información que recibimos del entorno. Para el problema de CartPole, recibimos la posición del palo, la velocidad y otros valores.

- Un **espacio de acción** que define las acciones posibles. En nuestro caso, el espacio de acción es discreto y consta de dos acciones: **izquierda** y **derecha**. (bloque de código 2)

1. Para inicializar, escribe el siguiente código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver cómo funciona el entorno, ejecutemos una simulación corta de 100 pasos. En cada paso, proporcionamos una de las acciones a realizar; en esta simulación simplemente seleccionamos una acción aleatoriamente del `action_space`.

1. Ejecuta el siguiente código y observa el resultado.

    ✅ Recuerda que es preferible ejecutar este código en una instalación local de Python. (bloque de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deberías ver algo similar a esta imagen:

    ![CartPole sin equilibrio](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante la simulación, necesitamos obtener observaciones para decidir cómo actuar. De hecho, la función `step` devuelve las observaciones actuales, una función de recompensa y una bandera `done` que indica si tiene sentido continuar la simulación o no: (bloque de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Verás algo como esto en la salida del notebook:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    El vector de observación que se devuelve en cada paso de la simulación contiene los siguientes valores:
    - Posición del carrito
    - Velocidad del carrito
    - Ángulo del palo
    - Tasa de rotación del palo

1. Obtén el valor mínimo y máximo de esos números: (bloque de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    También notarás que el valor de recompensa en cada paso de la simulación siempre es 1. Esto se debe a que nuestro objetivo es sobrevivir el mayor tiempo posible, es decir, mantener el palo en una posición razonablemente vertical durante el mayor tiempo posible.

    ✅ De hecho, la simulación de CartPole se considera resuelta si logramos obtener una recompensa promedio de 195 en 100 ensayos consecutivos.

## Discretización del estado

En Q-Learning, necesitamos construir una Q-Table que defina qué hacer en cada estado. Para poder hacer esto, el estado debe ser **discreto**, más precisamente, debe contener un número finito de valores discretos. Por lo tanto, necesitamos de alguna manera **discretizar** nuestras observaciones, mapeándolas a un conjunto finito de estados.

Hay algunas formas de hacer esto:

- **Dividir en intervalos**. Si conocemos el intervalo de un valor determinado, podemos dividir este intervalo en un número de **intervalos**, y luego reemplazar el valor por el número del intervalo al que pertenece. Esto se puede hacer utilizando el método [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) de numpy. En este caso, conoceremos con precisión el tamaño del estado, ya que dependerá del número de intervalos que seleccionemos para la digitalización.

✅ Podemos usar interpolación lineal para llevar los valores a un intervalo finito (por ejemplo, de -20 a 20), y luego convertir los números a enteros redondeándolos. Esto nos da un poco menos de control sobre el tamaño del estado, especialmente si no conocemos los rangos exactos de los valores de entrada. Por ejemplo, en nuestro caso, 2 de los 4 valores no tienen límites superiores/inferiores, lo que puede resultar en un número infinito de estados.

En nuestro ejemplo, utilizaremos el segundo enfoque. Como notarás más adelante, a pesar de los límites superiores/inferiores indefinidos, esos valores rara vez toman valores fuera de ciertos intervalos finitos, por lo que esos estados con valores extremos serán muy raros.

1. Aquí está la función que tomará la observación de nuestro modelo y producirá una tupla de 4 valores enteros: (bloque de código 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Exploremos también otro método de discretización utilizando intervalos: (bloque de código 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Ahora ejecutemos una simulación corta y observemos esos valores discretos del entorno. Siéntete libre de probar tanto `discretize` como `discretize_bins` y observa si hay alguna diferencia.

    ✅ `discretize_bins` devuelve el número del intervalo, que comienza en 0. Por lo tanto, para valores de la variable de entrada cercanos a 0, devuelve el número del medio del intervalo (10). En `discretize`, no nos preocupamos por el rango de los valores de salida, permitiendo que sean negativos, por lo que los valores del estado no están desplazados, y 0 corresponde a 0. (bloque de código 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Descomenta la línea que comienza con `env.render` si deseas ver cómo se ejecuta el entorno. De lo contrario, puedes ejecutarlo en segundo plano, lo cual es más rápido. Usaremos esta ejecución "invisible" durante nuestro proceso de Q-Learning.

## La estructura de la Q-Table

En nuestra lección anterior, el estado era un simple par de números del 0 al 8, por lo que era conveniente representar la Q-Table con un tensor de numpy con una forma de 8x8x2. Si usamos la discretización por intervalos, el tamaño de nuestro vector de estado también es conocido, por lo que podemos usar el mismo enfoque y representar el estado con un array de forma 20x20x10x10x2 (aquí 2 es la dimensión del espacio de acción, y las primeras dimensiones corresponden al número de intervalos que seleccionamos para cada uno de los parámetros en el espacio de observación).

Sin embargo, a veces las dimensiones precisas del espacio de observación no son conocidas. En el caso de la función `discretize`, nunca podemos estar seguros de que nuestro estado se mantenga dentro de ciertos límites, ya que algunos de los valores originales no están acotados. Por lo tanto, utilizaremos un enfoque ligeramente diferente y representaremos la Q-Table con un diccionario.

1. Usa el par *(estado, acción)* como clave del diccionario, y el valor corresponderá al valor de la entrada en la Q-Table. (bloque de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aquí también definimos una función `qvalues()`, que devuelve una lista de valores de la Q-Table para un estado dado que corresponde a todas las acciones posibles. Si la entrada no está presente en la Q-Table, devolveremos 0 como valor predeterminado.

## ¡Comencemos con Q-Learning!

Ahora estamos listos para enseñar a Pedro a mantener el equilibrio.

1. Primero, definamos algunos hiperparámetros: (bloque de código 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aquí, `alpha` es la **tasa de aprendizaje** que define en qué medida debemos ajustar los valores actuales de la Q-Table en cada paso. En la lección anterior comenzamos con 1 y luego disminuimos `alpha` a valores más bajos durante el entrenamiento. En este ejemplo lo mantendremos constante por simplicidad, pero puedes experimentar ajustando los valores de `alpha` más adelante.

    `gamma` es el **factor de descuento** que muestra en qué medida debemos priorizar la recompensa futura sobre la recompensa actual.

    `epsilon` es el **factor de exploración/explotación** que determina si debemos preferir la exploración o la explotación. En nuestro algoritmo, en un porcentaje `epsilon` de los casos seleccionaremos la siguiente acción según los valores de la Q-Table, y en el porcentaje restante ejecutaremos una acción aleatoria. Esto nos permitirá explorar áreas del espacio de búsqueda que nunca hemos visto antes.

    ✅ En términos de equilibrio, elegir una acción aleatoria (exploración) actuaría como un empujón aleatorio en la dirección equivocada, y el palo tendría que aprender a recuperar el equilibrio de esos "errores".

### Mejorar el algoritmo

Podemos hacer dos mejoras a nuestro algoritmo de la lección anterior:

- **Calcular la recompensa acumulativa promedio**, durante un número de simulaciones. Imprimiremos el progreso cada 5000 iteraciones, y promediaremos nuestra recompensa acumulativa durante ese período de tiempo. Esto significa que si obtenemos más de 195 puntos, podemos considerar el problema resuelto, con una calidad incluso mayor a la requerida.

- **Calcular el resultado acumulativo promedio máximo**, `Qmax`, y almacenaremos la Q-Table correspondiente a ese resultado. Cuando ejecutes el entrenamiento, notarás que a veces el resultado acumulativo promedio comienza a disminuir, y queremos conservar los valores de la Q-Table que corresponden al mejor modelo observado durante el entrenamiento.

1. Recopila todas las recompensas acumulativas en cada simulación en el vector `rewards` para su posterior representación gráfica. (bloque de código 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Lo que puedes notar de estos resultados:

- **Cerca de nuestro objetivo**. Estamos muy cerca de alcanzar el objetivo de obtener 195 recompensas acumulativas en más de 100 ejecuciones consecutivas de la simulación, ¡o incluso podemos haberlo logrado! Incluso si obtenemos números más bajos, no lo sabremos con certeza, ya que promediamos sobre 5000 ejecuciones, y solo se requieren 100 ejecuciones según el criterio formal.

- **La recompensa comienza a disminuir**. A veces la recompensa comienza a disminuir, lo que significa que podemos "destruir" valores ya aprendidos en la Q-Table con otros que empeoran la situación.

Esta observación es más clara si graficamos el progreso del entrenamiento.

## Graficar el progreso del entrenamiento

Durante el entrenamiento, hemos recopilado el valor de la recompensa acumulativa en cada una de las iteraciones en el vector `rewards`. Así es como se ve cuando lo graficamos contra el número de iteración:

```python
plt.plot(rewards)
```

![progreso sin procesar](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/images/train_progress_raw.png)

En este gráfico no es posible sacar conclusiones, ya que debido a la naturaleza del proceso de entrenamiento estocástico, la duración de las sesiones de entrenamiento varía mucho. Para darle más sentido a este gráfico, podemos calcular el **promedio móvil** sobre una serie de experimentos, digamos 100. Esto se puede hacer convenientemente usando `np.convolve`: (bloque de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progreso del entrenamiento](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variar los hiperparámetros

Para hacer el aprendizaje más estable, tiene sentido ajustar algunos de nuestros hiperparámetros durante el entrenamiento. En particular:

- **Para la tasa de aprendizaje**, `alpha`, podemos comenzar con valores cercanos a 1 y luego ir disminuyendo el parámetro. Con el tiempo, obtendremos buenos valores de probabilidad en la Q-Table, y por lo tanto deberíamos ajustarlos ligeramente, y no sobrescribirlos completamente con nuevos valores.

- **Aumentar epsilon**. Podríamos querer aumentar lentamente el valor de `epsilon`, para explorar menos y explotar más. Probablemente tenga sentido comenzar con un valor bajo de `epsilon` y aumentarlo hasta casi 1.
> **Tarea 1**: Prueba con diferentes valores de hiperparámetros y observa si puedes lograr una recompensa acumulativa más alta. ¿Estás obteniendo más de 195?
> **Tarea 2**: Para resolver formalmente el problema, necesitas alcanzar un promedio de recompensa de 195 a lo largo de 100 ejecuciones consecutivas. Mide eso durante el entrenamiento y asegúrate de que has resuelto el problema formalmente.

## Ver el resultado en acción

Sería interesante ver cómo se comporta el modelo entrenado. Vamos a ejecutar la simulación y seguir la misma estrategia de selección de acciones que durante el entrenamiento, muestreando según la distribución de probabilidad en la Q-Table: (bloque de código 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Deberías ver algo como esto:

![un carrito equilibrando](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafío

> **Tarea 3**: Aquí estuvimos utilizando la copia final de la Q-Table, que puede no ser la mejor. Recuerda que hemos almacenado la Q-Table con mejor rendimiento en la variable `Qbest`. ¡Prueba el mismo ejemplo con la Q-Table de mejor rendimiento copiando `Qbest` sobre `Q` y observa si notas alguna diferencia!

> **Tarea 4**: Aquí no estábamos seleccionando la mejor acción en cada paso, sino muestreando con la distribución de probabilidad correspondiente. ¿Tendría más sentido seleccionar siempre la mejor acción, con el valor más alto en la Q-Table? Esto se puede hacer utilizando la función `np.argmax` para encontrar el número de acción correspondiente al valor más alto en la Q-Table. Implementa esta estrategia y observa si mejora el equilibrio.

## [Cuestionario post-clase](https://ff-quizzes.netlify.app/en/ml/)

## Asignación
[Entrena un Mountain Car](assignment.md)

## Conclusión

Ahora hemos aprendido cómo entrenar agentes para lograr buenos resultados simplemente proporcionando una función de recompensa que define el estado deseado del juego y dándoles la oportunidad de explorar inteligentemente el espacio de búsqueda. Hemos aplicado con éxito el algoritmo de Q-Learning en casos de entornos discretos y continuos, pero con acciones discretas.

Es importante también estudiar situaciones donde el estado de las acciones sea continuo y cuando el espacio de observación sea mucho más complejo, como la imagen de la pantalla de un juego de Atari. En esos problemas, a menudo necesitamos usar técnicas de aprendizaje automático más poderosas, como redes neuronales, para lograr buenos resultados. Estos temas más avanzados serán el enfoque de nuestro próximo curso avanzado de IA.

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---


# Aplicaciones del Mundo Real

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5e069a0ac02a9606a69946c2b3c574a9",
  "translation_date": "2025-09-03T23:14:48+00:00",
  "source_file": "9-Real-World/README.md",
  "language_code": "es"
}
-->
# Posdata: Aplicaciones reales del aprendizaje automático clásico

En esta sección del currículo, se te presentarán algunas aplicaciones reales del aprendizaje automático clásico. Hemos investigado en internet para encontrar artículos y documentos técnicos sobre aplicaciones que han utilizado estas estrategias, evitando redes neuronales, aprendizaje profundo e inteligencia artificial tanto como sea posible. Aprende cómo se utiliza el aprendizaje automático en sistemas empresariales, aplicaciones ecológicas, finanzas, arte y cultura, entre otros.

![chess](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/translated_images/es/chess.e704a268781bdad8.webp)

> Foto por <a href="https://unsplash.com/@childeye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Alexis Fauvet</a> en <a href="https://unsplash.com/s/photos/artificial-intelligence?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Lección

1. [Aplicaciones reales del aprendizaje automático](1-Applications/README.md)
2. [Depuración de modelos en aprendizaje automático usando componentes del panel de Responsible AI](2-Debugging-ML-Models/README.md)

## Créditos

"Aplicaciones reales" fue escrito por un equipo de personas, incluyendo [Jen Looper](https://twitter.com/jenlooper) y [Ornella Altunyan](https://twitter.com/ornelladotcom).

"Depuración de modelos en aprendizaje automático usando componentes del panel de Responsible AI" fue escrito por [Ruth Yakubu](https://twitter.com/ruthieyakubu)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-04T22:19:09+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "es"
}
-->
# Posdata: Aprendizaje automático en el mundo real

![Resumen del aprendizaje automático en el mundo real en un sketchnote](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/sketchnotes/ml-realworld.png)  
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

En este plan de estudios, has aprendido muchas formas de preparar datos para el entrenamiento y crear modelos de aprendizaje automático. Construiste una serie de modelos clásicos de regresión, agrupamiento, clasificación, procesamiento de lenguaje natural y series temporales. ¡Felicidades! Ahora, podrías estar preguntándote para qué sirve todo esto... ¿cuáles son las aplicaciones reales de estos modelos?

Aunque la inteligencia artificial (IA), que generalmente utiliza aprendizaje profundo, ha captado mucho interés en la industria, todavía hay aplicaciones valiosas para los modelos clásicos de aprendizaje automático. ¡Incluso podrías estar usando algunas de estas aplicaciones hoy en día! En esta lección, explorarás cómo ocho industrias y dominios temáticos diferentes utilizan estos tipos de modelos para hacer que sus aplicaciones sean más eficientes, confiables, inteligentes y valiosas para los usuarios.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanzas

El sector financiero ofrece muchas oportunidades para el aprendizaje automático. Muchos problemas en esta área se prestan a ser modelados y resueltos utilizando ML.

### Detección de fraude con tarjetas de crédito

Aprendimos sobre [agrupamiento k-means](../../5-Clustering/2-K-Means/README.md) anteriormente en el curso, pero ¿cómo puede usarse para resolver problemas relacionados con el fraude con tarjetas de crédito?

El agrupamiento k-means es útil en una técnica de detección de fraude con tarjetas de crédito llamada **detección de valores atípicos**. Los valores atípicos, o desviaciones en las observaciones de un conjunto de datos, pueden indicarnos si una tarjeta de crédito se está utilizando de manera normal o si algo inusual está ocurriendo. Como se muestra en el artículo enlazado a continuación, puedes clasificar los datos de tarjetas de crédito utilizando un algoritmo de agrupamiento k-means y asignar cada transacción a un grupo según qué tan atípica parezca ser. Luego, puedes evaluar los grupos más riesgosos para determinar transacciones fraudulentas frente a legítimas.  
[Referencia](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestión de patrimonios

En la gestión de patrimonios, una persona o empresa maneja inversiones en nombre de sus clientes. Su trabajo es mantener y hacer crecer la riqueza a largo plazo, por lo que es esencial elegir inversiones que tengan un buen desempeño.

Una forma de evaluar cómo se desempeña una inversión en particular es a través de la regresión estadística. La [regresión lineal](../../2-Regression/1-Tools/README.md) es una herramienta valiosa para entender cómo se desempeña un fondo en relación con un punto de referencia. También podemos deducir si los resultados de la regresión son estadísticamente significativos o cuánto afectarían las inversiones de un cliente. Incluso podrías ampliar tu análisis utilizando regresión múltiple, donde se pueden tener en cuenta factores de riesgo adicionales. Para un ejemplo de cómo funcionaría esto para un fondo específico, consulta el artículo a continuación sobre la evaluación del desempeño de fondos utilizando regresión.  
[Referencia](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educación

El sector educativo también es un área muy interesante donde se puede aplicar ML. Hay problemas fascinantes por resolver, como detectar trampas en exámenes o ensayos, o gestionar sesgos, intencionados o no, en el proceso de corrección.

### Predicción del comportamiento estudiantil

[Coursera](https://coursera.com), un proveedor de cursos abiertos en línea, tiene un excelente blog técnico donde discuten muchas decisiones de ingeniería. En este estudio de caso, trazaron una línea de regresión para explorar cualquier correlación entre una baja calificación de NPS (Net Promoter Score) y la retención o abandono de cursos.  
[Referencia](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigación de sesgos

[Grammarly](https://grammarly.com), un asistente de escritura que revisa errores ortográficos y gramaticales, utiliza sofisticados [sistemas de procesamiento de lenguaje natural](../../6-NLP/README.md) en todos sus productos. Publicaron un interesante estudio de caso en su blog técnico sobre cómo abordaron el sesgo de género en el aprendizaje automático, algo que aprendiste en nuestra [lección introductoria sobre equidad](../../1-Introduction/3-fairness/README.md).  
[Referencia](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Comercio minorista

El sector minorista puede beneficiarse enormemente del uso de ML, desde crear una mejor experiencia para el cliente hasta optimizar el inventario.

### Personalización del recorrido del cliente

En Wayfair, una empresa que vende artículos para el hogar como muebles, ayudar a los clientes a encontrar los productos adecuados para sus gustos y necesidades es fundamental. En este artículo, los ingenieros de la empresa describen cómo utilizan ML y NLP para "mostrar los resultados correctos a los clientes". En particular, su motor de intención de consulta se ha construido para usar extracción de entidades, entrenamiento de clasificadores, extracción de opiniones y etiquetado de sentimientos en las reseñas de los clientes. Este es un caso clásico de cómo funciona el NLP en el comercio minorista en línea.  
[Referencia](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestión de inventarios

Empresas innovadoras y ágiles como [StitchFix](https://stitchfix.com), un servicio de cajas que envía ropa a los consumidores, dependen en gran medida de ML para recomendaciones y gestión de inventarios. Sus equipos de estilismo trabajan junto con sus equipos de comercialización: "uno de nuestros científicos de datos experimentó con un algoritmo genético y lo aplicó a prendas para predecir qué sería una pieza de ropa exitosa que aún no existe. Llevamos eso al equipo de comercialización y ahora pueden usarlo como una herramienta".  
[Referencia](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Salud

El sector de la salud puede aprovechar ML para optimizar tareas de investigación y también problemas logísticos como la readmisión de pacientes o la prevención de la propagación de enfermedades.

### Gestión de ensayos clínicos

La toxicidad en los ensayos clínicos es una gran preocupación para los fabricantes de medicamentos. ¿Cuánta toxicidad es tolerable? En este estudio, el análisis de varios métodos de ensayos clínicos llevó al desarrollo de un nuevo enfoque para predecir las probabilidades de los resultados de los ensayos clínicos. Específicamente, pudieron usar random forest para producir un [clasificador](../../4-Classification/README.md) capaz de distinguir entre grupos de medicamentos.  
[Referencia](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestión de readmisiones hospitalarias

La atención hospitalaria es costosa, especialmente cuando los pacientes deben ser readmitidos. Este artículo analiza una empresa que utiliza ML para predecir el potencial de readmisión utilizando algoritmos de [agrupamiento](../../5-Clustering/README.md). Estos grupos ayudan a los analistas a "descubrir grupos de readmisiones que pueden compartir una causa común".  
[Referencia](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestión de enfermedades

La reciente pandemia ha puesto de relieve cómo el aprendizaje automático puede ayudar a detener la propagación de enfermedades. En este artículo, reconocerás el uso de ARIMA, curvas logísticas, regresión lineal y SARIMA. "Este trabajo es un intento de calcular la tasa de propagación de este virus y, por lo tanto, predecir las muertes, recuperaciones y casos confirmados, para que podamos prepararnos mejor y sobrevivir".  
[Referencia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecología y tecnología verde

La naturaleza y la ecología consisten en muchos sistemas sensibles donde la interacción entre animales y la naturaleza es clave. Es importante medir estos sistemas con precisión y actuar adecuadamente si ocurre algo, como un incendio forestal o una disminución en la población animal.

### Gestión forestal

Aprendiste sobre [aprendizaje por refuerzo](../../8-Reinforcement/README.md) en lecciones anteriores. Puede ser muy útil al intentar predecir patrones en la naturaleza. En particular, puede usarse para rastrear problemas ecológicos como incendios forestales y la propagación de especies invasoras. En Canadá, un grupo de investigadores utilizó aprendizaje por refuerzo para construir modelos de dinámica de incendios forestales a partir de imágenes satelitales. Usando un innovador "proceso de propagación espacial (SSP)", imaginaron un incendio forestal como "el agente en cualquier celda del paisaje". "El conjunto de acciones que el fuego puede tomar desde una ubicación en cualquier momento incluye propagarse al norte, sur, este u oeste o no propagarse".  
[Referencia](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Detección de movimiento de animales

Aunque el aprendizaje profundo ha revolucionado el seguimiento visual de movimientos de animales (puedes construir tu propio [rastreador de osos polares](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aquí), el ML clásico todavía tiene un lugar en esta tarea.

Los sensores para rastrear movimientos de animales de granja e IoT hacen uso de este tipo de procesamiento visual, pero las técnicas más básicas de ML son útiles para preprocesar datos. Por ejemplo, en este artículo, se monitorearon y analizaron posturas de ovejas utilizando varios algoritmos clasificadores. Podrías reconocer la curva ROC en la página 335.  
[Referencia](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestión de energía

En nuestras lecciones sobre [pronóstico de series temporales](../../7-TimeSeries/README.md), invocamos el concepto de parquímetros inteligentes para generar ingresos para una ciudad basándonos en la comprensión de la oferta y la demanda. Este artículo analiza en detalle cómo el agrupamiento, la regresión y el pronóstico de series temporales se combinaron para ayudar a predecir el uso futuro de energía en Irlanda, basado en medidores inteligentes.  
[Referencia](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguros

El sector de seguros es otro sector que utiliza ML para construir y optimizar modelos financieros y actuariales viables.

### Gestión de volatilidad

MetLife, un proveedor de seguros de vida, es transparente sobre cómo analizan y mitigan la volatilidad en sus modelos financieros. En este artículo notarás visualizaciones de clasificación binaria y ordinal. También descubrirás visualizaciones de pronósticos.  
[Referencia](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Artes, cultura y literatura

En las artes, por ejemplo en el periodismo, hay muchos problemas interesantes. Detectar noticias falsas es un gran desafío, ya que se ha demostrado que influyen en la opinión de las personas e incluso pueden desestabilizar democracias. Los museos también pueden beneficiarse del uso de ML en todo, desde encontrar vínculos entre artefactos hasta la planificación de recursos.

### Detección de noticias falsas

Detectar noticias falsas se ha convertido en un juego del gato y el ratón en los medios de comunicación actuales. En este artículo, los investigadores sugieren que un sistema que combine varias de las técnicas de ML que hemos estudiado puede probarse y desplegarse el mejor modelo: "Este sistema se basa en el procesamiento de lenguaje natural para extraer características de los datos y luego estas características se utilizan para entrenar clasificadores de aprendizaje automático como Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) y Logistic Regression (LR)".  
[Referencia](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Este artículo muestra cómo combinar diferentes dominios de ML puede producir resultados interesantes que ayuden a detener la propagación de noticias falsas y evitar daños reales; en este caso, el impulso fue la propagación de rumores sobre tratamientos para el COVID que incitaron a la violencia.

### ML en museos

Los museos están al borde de una revolución de IA en la que catalogar y digitalizar colecciones y encontrar vínculos entre artefactos se está volviendo más fácil a medida que avanza la tecnología. Proyectos como [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) están ayudando a desbloquear los misterios de colecciones inaccesibles como los Archivos Vaticanos. Pero el aspecto comercial de los museos también se beneficia de los modelos de ML.

Por ejemplo, el Instituto de Arte de Chicago construyó modelos para predecir qué interesa a las audiencias y cuándo asistirán a exposiciones. El objetivo es crear experiencias de visitante individualizadas y optimizadas cada vez que el usuario visite el museo. "Durante el año fiscal 2017, el modelo predijo la asistencia y las admisiones con un 1% de precisión, dice Andrew Simnick, vicepresidente senior del Instituto de Arte".  
[Referencia](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentación de clientes

Las estrategias de marketing más efectivas apuntan a los clientes de diferentes maneras según varios grupos. En este artículo, se discuten los usos de los algoritmos de agrupamiento para apoyar el marketing diferenciado. El marketing diferenciado ayuda a las empresas a mejorar el reconocimiento de marca, llegar a más clientes y ganar más dinero.  
[Referencia](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Desafío

Identifica otro sector que se beneficie de algunas de las técnicas que aprendiste en este plan de estudios y descubre cómo utiliza ML.
## [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

El equipo de ciencia de datos de Wayfair tiene varios videos interesantes sobre cómo utilizan el aprendizaje automático en su empresa. Vale la pena [echarles un vistazo](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos).

## Tarea

[Una búsqueda del tesoro de aprendizaje automático](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.

---

<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-04T22:19:52+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "es"
}
-->
# Posdata: Depuración de modelos de aprendizaje automático utilizando componentes del panel de IA Responsable

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

El aprendizaje automático impacta nuestra vida cotidiana. La IA está encontrando su lugar en algunos de los sistemas más importantes que nos afectan como individuos y como sociedad, desde la atención médica, las finanzas, la educación y el empleo. Por ejemplo, los sistemas y modelos están involucrados en tareas de toma de decisiones diarias, como diagnósticos médicos o detección de fraudes. En consecuencia, los avances en IA junto con su adopción acelerada están siendo recibidos con expectativas sociales en evolución y una creciente regulación en respuesta. Constantemente vemos áreas donde los sistemas de IA no cumplen con las expectativas; exponen nuevos desafíos; y los gobiernos están comenzando a regular las soluciones de IA. Por lo tanto, es importante que estos modelos sean analizados para proporcionar resultados justos, confiables, inclusivos, transparentes y responsables para todos.

En este plan de estudios, exploraremos herramientas prácticas que pueden utilizarse para evaluar si un modelo tiene problemas relacionados con la IA Responsable. Las técnicas tradicionales de depuración de aprendizaje automático tienden a basarse en cálculos cuantitativos como la precisión agregada o la pérdida promedio de error. Imagina lo que puede suceder cuando los datos que estás utilizando para construir estos modelos carecen de ciertos grupos demográficos, como raza, género, visión política, religión, o representan desproporcionadamente dichos grupos demográficos. ¿Qué sucede cuando la salida del modelo se interpreta como favorable para algún grupo demográfico? Esto puede introducir una representación excesiva o insuficiente de estos grupos sensibles, lo que resulta en problemas de equidad, inclusión o confiabilidad del modelo. Otro factor es que los modelos de aprendizaje automático son considerados cajas negras, lo que dificulta entender y explicar qué impulsa las predicciones de un modelo. Todos estos son desafíos que enfrentan los científicos de datos y desarrolladores de IA cuando no tienen herramientas adecuadas para depurar y evaluar la equidad o confiabilidad de un modelo.

En esta lección, aprenderás a depurar tus modelos utilizando:

- **Análisis de errores**: identificar dónde en la distribución de tus datos el modelo tiene altas tasas de error.
- **Visión general del modelo**: realizar análisis comparativos entre diferentes cohortes de datos para descubrir disparidades en las métricas de rendimiento de tu modelo.
- **Análisis de datos**: investigar dónde podría haber una representación excesiva o insuficiente de tus datos que pueda sesgar tu modelo para favorecer un grupo demográfico sobre otro.
- **Importancia de las características**: comprender qué características están impulsando las predicciones de tu modelo a nivel global o local.

## Prerrequisito

Como prerrequisito, revisa [Herramientas de IA Responsable para desarrolladores](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif sobre herramientas de IA Responsable](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Análisis de errores

Las métricas tradicionales de rendimiento de modelos utilizadas para medir la precisión son principalmente cálculos basados en predicciones correctas frente a incorrectas. Por ejemplo, determinar que un modelo es preciso el 89% del tiempo con una pérdida de error de 0.001 puede considerarse un buen rendimiento. Los errores a menudo no se distribuyen uniformemente en tu conjunto de datos subyacente. Puedes obtener una puntuación de precisión del modelo del 89%, pero descubrir que hay diferentes regiones de tus datos en las que el modelo falla el 42% del tiempo. La consecuencia de estos patrones de falla con ciertos grupos de datos puede llevar a problemas de equidad o confiabilidad. Es esencial comprender las áreas donde el modelo está funcionando bien o no. Las regiones de datos donde hay un alto número de inexactitudes en tu modelo pueden resultar ser un grupo demográfico importante.

![Analizar y depurar errores del modelo](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

El componente de Análisis de Errores en el panel de IA Responsable ilustra cómo se distribuyen las fallas del modelo entre varios cohortes con una visualización en forma de árbol. Esto es útil para identificar características o áreas donde hay una alta tasa de error en tu conjunto de datos. Al observar de dónde provienen la mayoría de las inexactitudes del modelo, puedes comenzar a investigar la causa raíz. También puedes crear cohortes de datos para realizar análisis. Estos cohortes de datos ayudan en el proceso de depuración para determinar por qué el rendimiento del modelo es bueno en un cohorte pero erróneo en otro.

![Análisis de errores](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Los indicadores visuales en el mapa del árbol ayudan a localizar las áreas problemáticas más rápidamente. Por ejemplo, cuanto más oscuro sea el tono de rojo en un nodo del árbol, mayor será la tasa de error.

El mapa de calor es otra funcionalidad de visualización que los usuarios pueden utilizar para investigar la tasa de error utilizando una o dos características para encontrar un contribuyente a los errores del modelo en todo el conjunto de datos o cohortes.

![Mapa de calor de análisis de errores](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Utiliza el análisis de errores cuando necesites:

* Obtener una comprensión profunda de cómo se distribuyen las fallas del modelo en un conjunto de datos y en varias dimensiones de entrada y características.
* Desglosar las métricas de rendimiento agregadas para descubrir automáticamente cohortes erróneos que informen tus pasos de mitigación específicos.

## Visión general del modelo

Evaluar el rendimiento de un modelo de aprendizaje automático requiere obtener una comprensión holística de su comportamiento. Esto puede lograrse revisando más de una métrica, como tasa de error, precisión, recall, precisión o MAE (Error Absoluto Medio) para encontrar disparidades entre las métricas de rendimiento. Una métrica de rendimiento puede parecer excelente, pero las inexactitudes pueden exponerse en otra métrica. Además, comparar las métricas para encontrar disparidades en todo el conjunto de datos o cohortes ayuda a arrojar luz sobre dónde el modelo está funcionando bien o no. Esto es especialmente importante para observar el rendimiento del modelo entre características sensibles frente a insensibles (por ejemplo, raza, género o edad del paciente) para descubrir posibles problemas de equidad que pueda tener el modelo. Por ejemplo, descubrir que el modelo es más erróneo en un cohorte que tiene características sensibles puede revelar posibles problemas de equidad en el modelo.

El componente de Visión General del Modelo del panel de IA Responsable ayuda no solo a analizar las métricas de rendimiento de la representación de datos en un cohorte, sino que también brinda a los usuarios la capacidad de comparar el comportamiento del modelo entre diferentes cohortes.

![Cohortes de conjuntos de datos - visión general del modelo en el panel de IA Responsable](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

La funcionalidad de análisis basada en características del componente permite a los usuarios reducir subgrupos de datos dentro de una característica particular para identificar anomalías a nivel granular. Por ejemplo, el panel tiene inteligencia integrada para generar automáticamente cohortes para una característica seleccionada por el usuario (por ejemplo, *"time_in_hospital < 3"* o *"time_in_hospital >= 7"*). Esto permite al usuario aislar una característica particular de un grupo de datos más grande para ver si es un factor clave en los resultados erróneos del modelo.

![Cohortes de características - visión general del modelo en el panel de IA Responsable](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

El componente de Visión General del Modelo admite dos clases de métricas de disparidad:

**Disparidad en el rendimiento del modelo**: Este conjunto de métricas calcula la disparidad (diferencia) en los valores de la métrica de rendimiento seleccionada entre subgrupos de datos. Aquí hay algunos ejemplos:

* Disparidad en la tasa de precisión
* Disparidad en la tasa de error
* Disparidad en la precisión
* Disparidad en el recall
* Disparidad en el error absoluto medio (MAE)

**Disparidad en la tasa de selección**: Esta métrica contiene la diferencia en la tasa de selección (predicción favorable) entre subgrupos. Un ejemplo de esto es la disparidad en las tasas de aprobación de préstamos. La tasa de selección significa la fracción de puntos de datos en cada clase clasificados como 1 (en clasificación binaria) o la distribución de valores de predicción (en regresión).

## Análisis de datos

> "Si torturas los datos lo suficiente, confesarán cualquier cosa" - Ronald Coase

Esta afirmación suena extrema, pero es cierto que los datos pueden manipularse para respaldar cualquier conclusión. Tal manipulación a veces puede ocurrir de manera no intencional. Como humanos, todos tenemos sesgos, y a menudo es difícil saber conscientemente cuándo estás introduciendo sesgos en los datos. Garantizar la equidad en la IA y el aprendizaje automático sigue siendo un desafío complejo.

Los datos son un gran punto ciego para las métricas tradicionales de rendimiento de modelos. Puedes tener puntuaciones de precisión altas, pero esto no siempre refleja el sesgo subyacente en los datos que podría estar en tu conjunto de datos. Por ejemplo, si un conjunto de datos de empleados tiene un 27% de mujeres en puestos ejecutivos en una empresa y un 73% de hombres en el mismo nivel, un modelo de IA para publicidad de empleo entrenado con estos datos puede dirigirse principalmente a una audiencia masculina para puestos de alto nivel. Tener este desequilibrio en los datos sesgó la predicción del modelo para favorecer un género. Esto revela un problema de equidad donde hay un sesgo de género en el modelo de IA.

El componente de Análisis de Datos en el panel de IA Responsable ayuda a identificar áreas donde hay una representación excesiva o insuficiente en el conjunto de datos. Ayuda a los usuarios a diagnosticar la causa raíz de errores y problemas de equidad introducidos por desequilibrios en los datos o la falta de representación de un grupo de datos particular. Esto brinda a los usuarios la capacidad de visualizar conjuntos de datos basados en resultados predichos y reales, grupos de errores y características específicas. A veces, descubrir un grupo de datos subrepresentado también puede revelar que el modelo no está aprendiendo bien, de ahí las altas inexactitudes. Tener un modelo con sesgo en los datos no solo es un problema de equidad, sino que muestra que el modelo no es inclusivo ni confiable.

![Componente de análisis de datos en el panel de IA Responsable](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Utiliza el análisis de datos cuando necesites:

* Explorar las estadísticas de tu conjunto de datos seleccionando diferentes filtros para dividir tus datos en diferentes dimensiones (también conocidas como cohortes).
* Comprender la distribución de tu conjunto de datos entre diferentes cohortes y grupos de características.
* Determinar si tus hallazgos relacionados con equidad, análisis de errores y causalidad (derivados de otros componentes del panel) son resultado de la distribución de tu conjunto de datos.
* Decidir en qué áreas recolectar más datos para mitigar errores que provienen de problemas de representación, ruido en las etiquetas, ruido en las características, sesgo en las etiquetas y factores similares.

## Interpretabilidad del modelo

Los modelos de aprendizaje automático tienden a ser cajas negras. Comprender qué características clave de los datos impulsan la predicción de un modelo puede ser un desafío. Es importante proporcionar transparencia sobre por qué un modelo hace una determinada predicción. Por ejemplo, si un sistema de IA predice que un paciente diabético está en riesgo de ser readmitido en un hospital en menos de 30 días, debería poder proporcionar datos de apoyo que llevaron a su predicción. Tener indicadores de datos de apoyo aporta transparencia para ayudar a los médicos o hospitales a tomar decisiones bien informadas. Además, poder explicar por qué un modelo hizo una predicción para un paciente individual permite responsabilidad con las regulaciones de salud. Cuando utilizas modelos de aprendizaje automático de maneras que afectan la vida de las personas, es crucial comprender y explicar qué influye en el comportamiento de un modelo. La explicabilidad e interpretabilidad del modelo ayuda a responder preguntas en escenarios como:

* Depuración del modelo: ¿Por qué mi modelo cometió este error? ¿Cómo puedo mejorar mi modelo?
* Colaboración humano-IA: ¿Cómo puedo entender y confiar en las decisiones del modelo?
* Cumplimiento normativo: ¿Cumple mi modelo con los requisitos legales?

El componente de Importancia de las Características del panel de IA Responsable te ayuda a depurar y obtener una comprensión integral de cómo un modelo hace predicciones. También es una herramienta útil para profesionales de aprendizaje automático y tomadores de decisiones para explicar y mostrar evidencia de las características que influyen en el comportamiento de un modelo para el cumplimiento normativo. Los usuarios pueden explorar explicaciones globales y locales para validar qué características impulsan la predicción de un modelo. Las explicaciones globales enumeran las principales características que afectaron la predicción general de un modelo. Las explicaciones locales muestran qué características llevaron a la predicción de un modelo para un caso individual. La capacidad de evaluar explicaciones locales también es útil para depurar o auditar un caso específico para comprender mejor y explicar por qué un modelo hizo una predicción precisa o inexacta.

![Componente de importancia de características en el panel de IA Responsable](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Explicaciones globales: Por ejemplo, ¿qué características afectan el comportamiento general de un modelo de readmisión hospitalaria para pacientes diabéticos?
* Explicaciones locales: Por ejemplo, ¿por qué se predijo que un paciente diabético mayor de 60 años con hospitalizaciones previas sería readmitido o no readmitido dentro de los 30 días en un hospital?

En el proceso de depuración al examinar el rendimiento de un modelo entre diferentes cohortes, la Importancia de las Características muestra qué nivel de impacto tiene una característica en los cohortes. Ayuda a revelar anomalías al comparar el nivel de influencia que tiene la característica en impulsar las predicciones erróneas de un modelo. El componente de Importancia de las Características puede mostrar qué valores en una característica influyeron positiva o negativamente en el resultado del modelo. Por ejemplo, si un modelo hizo una predicción inexacta, el componente te da la capacidad de profundizar y señalar qué características o valores de características impulsaron la predicción. Este nivel de detalle no solo ayuda en la depuración, sino que proporciona transparencia y responsabilidad en situaciones de auditoría. Finalmente, el componente puede ayudarte a identificar problemas de equidad. Para ilustrar, si una característica sensible como etnia o género tiene una alta influencia en impulsar la predicción de un modelo, esto podría ser un indicio de sesgo racial o de género en el modelo.

![Importancia de características](https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Utiliza la interpretabilidad cuando necesites:

* Determinar qué tan confiables son las predicciones de tu sistema de IA al comprender qué características son más importantes para las predicciones.
* Abordar la depuración de tu modelo al comprenderlo primero e identificar si el modelo está utilizando características saludables o simplemente correlaciones falsas.
* Descubrir posibles fuentes de inequidad al comprender si el modelo está basando sus predicciones en características sensibles o en características altamente correlacionadas con ellas.
* Generar confianza en las decisiones de tu modelo al generar explicaciones locales para ilustrar sus resultados.
* Completar una auditoría regulatoria de un sistema de IA para validar modelos y monitorear el impacto de las decisiones del modelo en las personas.

## Conclusión

Todos los componentes del panel de IA Responsable son herramientas prácticas para ayudarte a construir modelos de aprendizaje automático que sean menos perjudiciales y más confiables para la sociedad. Mejoran la prevención de amenazas a los derechos humanos; la discriminación o exclusión de ciertos grupos de oportunidades de vida; y el riesgo de daño físico o psicológico. También ayudan a generar confianza en las decisiones de tu modelo al generar explicaciones locales para ilustrar sus resultados. Algunos de los posibles daños pueden clasificarse como:

- **Asignación**, si un género o etnia, por ejemplo, es favorecido sobre otro.
- **Calidad del servicio**. Si entrenas los datos para un escenario específico pero la realidad es mucho más compleja, esto lleva a un servicio de bajo rendimiento.
- **Estereotipos**. Asociar un grupo dado con atributos preasignados.
- **Denigración**. Criticar injustamente y etiquetar algo o alguien.
- **Sobre- o sub-representación**. La idea es que un cierto grupo no se vea en una determinada profesión, y cualquier servicio o función que siga promoviendo eso está contribuyendo al daño.

### Azure RAI dashboard

[Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) está construido sobre herramientas de código abierto desarrolladas por instituciones académicas y organizaciones líderes, incluyendo Microsoft. Estas herramientas son fundamentales para que los científicos de datos y desarrolladores de IA comprendan mejor el comportamiento de los modelos, descubran y mitiguen problemas indeseables en los modelos de IA.

- Aprende cómo usar los diferentes componentes consultando la [documentación del RAI dashboard.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Consulta algunos [notebooks de ejemplo del RAI dashboard](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) para depurar escenarios de IA más responsables en Azure Machine Learning.

---
## 🚀 Desafío

Para evitar que se introduzcan sesgos estadísticos o de datos desde el principio, deberíamos:

- contar con una diversidad de antecedentes y perspectivas entre las personas que trabajan en los sistemas
- invertir en conjuntos de datos que reflejen la diversidad de nuestra sociedad
- desarrollar mejores métodos para detectar y corregir sesgos cuando ocurren

Piensa en escenarios de la vida real donde la injusticia sea evidente en la construcción y uso de modelos. ¿Qué más deberíamos considerar?

## [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)
## Revisión y Autoestudio

En esta lección, has aprendido algunas herramientas prácticas para incorporar IA responsable en el aprendizaje automático.

Mira este taller para profundizar en los temas:

- Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica por Besmira Nushi y Mehrnoosh Sameki

[![Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica")


> 🎥 Haz clic en la imagen de arriba para ver el video: Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica por Besmira Nushi y Mehrnoosh Sameki

Consulta los siguientes materiales para aprender más sobre IA responsable y cómo construir modelos más confiables:

- Herramientas del RAI dashboard de Microsoft para depurar modelos de aprendizaje automático: [Recursos de herramientas de IA responsable](https://aka.ms/rai-dashboard)

- Explora el kit de herramientas de IA responsable: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Centro de recursos de IA responsable de Microsoft: [Recursos de IA Responsable – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de investigación FATE de Microsoft: [FATE: Equidad, Responsabilidad, Transparencia y Ética en IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Tarea

[Explora el RAI dashboard](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.

---
