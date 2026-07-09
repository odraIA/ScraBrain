# Guion de defensa - 15 minutos

## 0. Portada - 0:00-0:20

Buenas, soy Ricardo Díaz Peris y voy a presentar mi TFM, titulado *Transferencia de modelos MEG de contexto largo a EEG para clasificación contextual de palabras*. La idea central es estudiar si una arquitectura aprendida con MEG puede servir como punto de partida para EEG, que es más ruidoso pero mucho más viable para aplicaciones portables.

## 1. ¿Cómo de difícil es leer la mente? - 0:20-1:15

¿Cómo de difícil es leer la mente? Pongo este título para sumar sensacionalismo pero realmente el objetivo no es leer pensamientos de forma directa. Lo que se intenta aprender es una relación entre señales cerebrales y estímulos lingüísticos. El objetivo a largo plazo de las BCI y del brain-to-text es traducir señales cerebrales en lenguaje útil para comunicación, pero eso sigue siendo difícil por varios motivos. La señal cerebral no invasiva tiene mucho ruido, y la señal de una palabra no sucede solo durante el tiempo que se dice esa palabra. Sumado a esto, los datos que hay son bastante escasos y MUy diferentes entre usuarios, tú y tú tenéis diferentes señales cerebrales con respecto a la misma palabra y eso es una cosa que el modelo debe aprender.

## 2. Intrusivo vs. no intrusivo - 1:15-2:00

El origen y motivación del proyecto es que Vicent vio este video en el que esta chica, Ann, consigue volver a hablar utilizando un método invasivo llamado electrocorticografía, que es un método invasivo. Y es que los métodos para detectar señales cerebrales se dividen en 2 clases, invasivos y no invasivos. Los métodos invasivos (como ECoG) requieren cirugía pero suelen ofrecer mejor señal. Eso los hace muy potentes en contextos clínicos concretos al estar tan cerca del cerebro, pero poco escalables. En cambio, el electroencefalograma (EEG), magnetoencefalograma (MEG) y resonancia magnética funcional (fMRI) son no invasivos. Se puede ver fácilmente cuál de las 3 es más portátil, y yo por ese momento lo único que sabía de EEG es esta chica, Perri Karyal, que la usa para jugar a videojuegos sin utilizar el mando.

## 3. Comparativa de métodos no intrusivos - 2:00-2:45

Entonces comenzamos a investigar y comparar los diferentes métodos no invasivos. La tabla lo resume a grandes rasgos. Hay 2 cosas que queremos mirar con detalle, la resolución espacial (dónde ha sucedido en el cerebro) y la resolución temporal (cuándo ha sucedido). EEG y MEG tienen alta resolución temporal, que es importante para lenguaje porque las palabras ocurren en escalas de milisegundos. fMRI tiene buena resolución espacial, pero peor resolución temporal. El TFM se centra en EEG y MEG porque el objetivo es clasificar palabras alineadas en el tiempo, no reconstruir semántica global a partir de imágenes cerebrales lentas.

## 4. MEG vs EEG - 2:45-3:30

Una vez decidimos centrarnos en EEG y MEG, tuvimos que entenderlas a fondo. Las dos modalidades observan actividad neuronal relacionada, pero no la observan igual. EEG mide diferencias de potenciales eléctricos que llegan al cuero cabelludo, mientras que MEG mide campos magnéticos con direcciones desde sensores externos. Es decir, la fuente neuronal puede ser la misma o estar relacionada, pero la forma en la que llega al sensor cambia mucho.

Por el tema del coste y poder reproducir experimentos la opción que más sencillo y barato sería para futuros usuarios es EEG. Por lo que decidimos escoger esa. EEG se ha utilizado hasta ahora para movilidad y emociones, pero casi no para lenguaje.

Aun así, MEG es importante para nuestro trabajo ya que no basta con coger un modelo entrenado en MEG y aplicarlo directamente sobre EEG. La arquitectura puede servir como punto de partida, pero la entrada tiene que adaptarse: cambian los sensores, cambia la geometría, cambia el ruido y cambia la forma en la que la señal se proyecta hacia fuera.

Y como el objetivo final es EEG, antes de hablar del modelo quiero aterrizar un poco qué significa medir con un casco EEG real.

## 5. Qué capta realmente un casco de EEG - 3:30-4:15

Un casco de EEG no mide pensamientos. Mide diferencias de potencial muy pequeñas en el cuero cabelludo, producidas por actividad neuronal sincronizada y registradas mediante electrodos. El problema es que esa señal útil llega mezclada con muchísimas cosas que no nos interesan: parpadeos, movimiento ocular, tensión muscular, mala impedancia, movimiento del casco y ruido eléctrico.

Por eso esta diapositiva es importante. Cuando hablamos de pasar de señales cerebrales a palabras, no estamos partiendo de una señal limpia y directa. Estamos partiendo de microvoltios, muy contaminados y muy variables entre personas y sesiones. De ahí salen tres necesidades del trabajo: preprocesar bien, usar contexto largo y aprovechar preentrenamiento.

## 6. Momento casco - 4:15-4:55

Aquí tenéis una foto para verlo bien. Estos electrodos son los que contactan con el cuero cabelludo y registran diferencias de potencial. Lo importante es que están fuera del cráneo, de forma que es como decía no invasivo, impreso en 3d es este que es más barato en comparación con MEG y bastante más portable. Pero esa misma ventaja es también su limitación: la señal llega atenuada, mezclada y con mucho ruido.

Por eso no conviene vender EEG como si fuera una lectura directa del cerebro. Lo que hacemos en este trabajo es mucho más concreto: entrenar un modelo para encontrar patrones estadísticos entre EEG y palabras en una tarea experimental controlada.

*Enseño el casco, señalo electrodos y vuelvo rápido a la presentación. Si puedo llegar a conectar el casco y enseñar el software molaría mucho pero no sé si tengo tiempo

Con esa idea clara, ya puedo formular la pregunta concreta del TFM.

## 7. Pregunta de investigación - 4:55-5:35

Hay tres piezas. La primera es MEG-XL, que es el punto de partida: una arquitectura autosupervisada que utiliza contexto largo pensada para señales MEG. A partir de aquí hemos creado EEG-XL, que es la adaptación que permite trabajar con EEG, incorporando posiciones, máscaras y tipo de sensor aplicado a EEG. Y la tercera es la evaluación, que se hace como recuperación Top-10 de palabras en OpenNeuro ds004408.

Teniendo esta base la pregunta central es esta: ¿puede un modelo MEG de contexto largo adaptarse a EEG para clasificación contextual de palabras? 

Aquí quiero dejar claro que cuando hablo de palabras me refiero a palabras, no frases. No estoy diciendo que el modelo escuche una señal cerebral y escriba una frase entera. La tarea es más controlada: para cada posición, se evalúa si la palabra correcta aparece entre las diez candidatas mejor puntuadas.

A partir de esta pregunta, la propuesta consiste en conservar lo que hace fuerte a MEG-XL, pero cambiando lo necesario para que pueda funcionar con EEG.

## 8. Propuesta: de MEG-XL a EEG-XL - 5:35-6:35

Como he dicho antes nos basamos en MEG-XL, que nace con el objetivo de ser un modelo de pocos parámetros que sea capaz de entender el lenguje en MEG y a partir de ahí ya especialiarlo en tareas específicas. Tiene dos ideas que interesan mucho para lenguaje: trabaja con contexto largo y aprende de forma autosupervisada. Eso encaja bien con nuestro problema, porque una palabra no depende solo del instante exacto en el que aparece, sino del contexto temporal y lingüístico en el que ocurre.

Primero partimos de señal EEG continua y la organizamos en ventanas de 150 segundos. Después BioCodec (tokenizador orignialmente de EEG) discretiza la señal por canal, de forma que el modelo trabaja con tokens neuronales. Luego se añade información del sensor: posición 3D, tipo de sensor y máscara, porque no todos los datasets tienen los mismos canales ni la misma disposición. A continuación entra el Transformer criss-cross, que modela tanto la dimensión temporal como la dimensión espacial de sensores. Finalmente, en el fine-tuning, esa representación se usa para hacer ranking de palabras mediante embeddings de T5.

El cambio clave respecto a MEG no es solo cambiar el nombre de la modalidad. Es hacer que la entrada acepte EEG heterogéneo: diferentes cascos, diferentes canales, sensores ausentes y sesiones distintas.

Para que este modelo aprenda algo útil, el siguiente problema es con qué datos entrenarlo y cómo ordenar esos datos.

## 9. Datasets: lectura para adaptar, escucha para acercarse a la tarea final - 6:35-7:25

Dediqué mucho tiempo buscando datasets ya que no existen casi datasets de escucha continua en EEG. De hecho, hace nada han sacado un benchmark de EEG en el que NO se incluye nada de percepción del habla. El entrenamiento se organiza como una progresión. Primero uso datasets de lectura, como ZuCo 2.0 y Nieuwland. No son exactamente la misma tarea final, pero ya aportan al modelo una primera idea de EEG relacionado con procesamiento lingüístico y permiten una adaptación inicial de la arquitectura a señales EEG.

Después paso a datasets de escucha, como SparrKULee y OpenNeuro ds007808, porque se acercan más al escenario de palabras habladas y a la evaluación final. Y dejo OpenNeuro ds004408 separado para fine-tuning y test. Esto es importante metodológicamente: ds004408 no participa en el preentrenamiento, para evitar contaminar la evaluación final.

La idea no es decir que todos los datasets sean perfectamente equivalentes. De hecho, no lo son. La idea es aprovechar lo que hay disponible y ordenar el entrenamiento de lo más general a lo más parecido a la tarea final.

Como estos datasets son distintos entre sí, antes de meterlos al modelo hace falta convertirlos a una representación común.

## 10. Preprocesamiento común - 7:25-8:10

El objetivo del preprocesamiento es que señales EEG de datasets muy distintos acaben en una entrada compatible. 

Aquí voy a estar hablando de 2 frecuencias distintas. Una es la frecuencia de la propia señal EEG: las oscilaciones que queremos conservar, por eso aparece el filtrado 0,1-40 Hz. La otra es la frecuencia de muestreo: cuántas muestras guardamos por segundo, que después del remuestreo es 50 Hz. Por Nyquist, si muestreo a 50 Hz, la frecuencia máxima representable sin ambigüedad es la mitad: 25 Hz. Por eso, aunque el filtro inicial llegue a 40 Hz, en la práctica la banda efectiva del modelo queda limitada aproximadamente a 0,1-25 Hz.

Esto es importante porque condiciona qué información puede utilizar realmente el modelo. Es una decisión de compromiso: se pierde parte de la banda alta, pero se reduce mucho el coste computacional y se hace viable trabajar con ventanas largas.

Remuestreada la señal a 50 Hz, se construyen ventanas completas de 150 segundos y se incorporan las posiciones 3D y las máscaras de sensores. La señal se recorta en bloques completos: si al final queda un fragmento más corto que 150 segundos, no se usa.

Con las señales ya normalizadas, el primer entrenamiento no usa todavía palabras. Primero se aprende a modelar la propia señal EEG.

## 11. Preentrenamiento autosupervisado - 8:10-9:05

En esta fase el modelo aprende la estructura de la señal cerebral, no palabras. BioCodec convierte cada canal en tokens discretos, y luego el modelo recibe una ventana larga de 150 segundos dividida en bloques de 3 segundos. De esos 50 bloques, se enmascara aproximadamente el 40 %, es decir, 20 bloques, y el Transformer tiene que reconstruir los tokens ocultos usando el contexto que sí ve.

La ventaja de esto es que no necesito etiquetas lingüísticas para todo. El modelo puede aprender regularidades generales del EEG: patrones temporales, relaciones entre sensores y estructura de la señal. Después, cuando ya pase al fine-tuning con palabras, no parte de una representación aleatoria.

Aquí la idea de contexto largo es clave. Si solo miramos ventanas muy cortas, el modelo aprende patrones locales. Con 150 segundos puede explotar información más estable de la sesión, del sujeto y del contexto temporal.

Una vez el modelo ha aprendido una representación general de EEG, pasamos a la tarea supervisada: recuperar palabras.

## 12. Fine-tuning: recuperación contextual de palabras - 9:05-9:55

En el fine-tuning ya usamos la alineación entre señal EEG y palabras. Cada palabra tiene un tiempo de inicio, extraído de las anotaciones. A partir de ese instante se toma una ventana EEG de 3 segundos (0,5s antes de que comience la palabra y 2.5s después). Después se agrupan 50 palabras consecutivas, lo que vuelve a formar una entrada de 150 segundos.

La salida del modelo se proyecta a un espacio de 1024 dimensiones, que corresponde a embeddings de T5-large. La evaluación se hace por similitud coseno: para cada palabra, el modelo genera una representación y se comprueba si la palabra correcta está entre las 10 más cercanas dentro del conjunto de candidatas.

Esto es lo que llamo clasificación contextual de palabras. No se clasifica una palabra aislada sin contexto, sino una secuencia de palabras, de forma que el Transformer puede usar información temporal alrededor de cada posición.

Para saber qué parte de la mejora viene de cada decisión, planteé cuatro condiciones experimentales.

## 13. Diseño experimental: separar efectos - 9:55-10:50

Esta tabla es importante porque organiza la comparación. No quería entrenar un único modelo y decir simplemente que funciona. La idea era separar efectos.

El primer experimento es sin preentrenamiento: inicialización aleatoria y fine-tuning directo en ds004408. Esto funciona como control. El segundo es EEG desde cero: también empieza aleatoria, pero sí pasa por el preentrenamiento EEG con lectura y escucha. Así se mide cuánto aporta el preentrenamiento autosupervisado.

Los otros 2 experimentos parten del checkpoint de MEG-XL. En una redefino el embedding como EEG, y en la otra reutilizo el embedding MEG. Esto permite medir dos cosas: si inicializar desde MEG-XL ayuda y si conviene decirle explícitamente al modelo que la modalidad ahora es EEG o mantener el embedding original de MEG.

La gracia del diseño es que cada condición responde a una pregunta distinta: qué pasa sin preentrenar, qué aporta preentrenar en EEG, qué añade MEG-XL y cómo afecta el tipo de sensor.

Con este diseño, el resultado principal se resume en la siguiente tabla.

## 14. Resultado principal - 10:50-12:05

La métrica principal es Top-10 balanceada sobre las 250 palabras más frecuentes de ds004408. Con 250 candidatas, el azar uniforme sería un 4 %. Esto nos da una referencia clara para interpretar los resultados.

El modelo sin preentrenamiento obtiene 4,05 %, prácticamente azar. Esto es importante porque muestra que el fine-tuning directo con EEG no basta. Cuando añadimos preentrenamiento EEG desde cero, el resultado sube a 19,95 %, casi cinco veces el azar. Ese es el salto grande del trabajo.

Después, al inicializar desde MEG-XL y usar embedding EEG, se alcanza 20,56 %. Y el mejor resultado aparece con MEG-XL reutilizando el embedding MEG, con 22,32 %, que equivale a 5,58 veces el azar.

La lectura rápida es esta: el preentrenamiento EEG es el factor decisivo. La transferencia desde MEG-XL también ayuda, pero la ganancia es más moderada. Por tanto, no vendería el resultado como que MEG resuelve EEG, sino como que MEG aporta una inicialización útil encima de un preentrenamiento EEG que ya es fundamental.

Además del resultado final, miré cómo evoluciona la validación durante el fine-tuning.

## 15. Dinámica de validación - 12:05-12:50

Esta gráfica muestra que los mejores checkpoints aparecen bastante pronto. La curva sube rápido y después no necesariamente mejora. En algunos casos incluso puede degradarse. Esto encaja con lo que esperaría en EEG: los datos son ruidosos, el conjunto supervisado es limitado y el modelo puede sobreajustar con facilidad.

La interpretación es que el preentrenamiento proporciona una representación útil, pero el fine-tuning tiene que controlarse bien. No es una situación en la que entrenar más épocas garantice mejores resultados. Aquí elegir el checkpoint correcto y validar con cuidado es parte importante del protocolo experimental.

Con esto, la interpretación general debe ser positiva, pero prudente.

## 16. Interpretación de los resultados - 12:50-13:45

La interpretación honesta tiene tres partes. Primero, hay viabilidad: la arquitectura se ha podido adaptar a EEG y queda claramente por encima del azar en una tarea de recuperación de palabras. Además, el contexto largo parece compatible con este tipo de evaluación.

Segundo, hay una lectura sobre qué aporta más. El mayor salto viene del preentrenamiento EEG. MEG-XL añade una mejora adicional, y en esta ejecución el mejor resultado aparece reutilizando el embedding MEG. Eso sugiere que hay información transferible, aunque no sea el único factor ni el más grande.

Tercero, hay que ser prudente. Es una única semilla, los protocolos no son idénticos a todos los trabajos previos y no estamos haciendo generación libre de texto. Por tanto, el resultado no significa que el brain-to-text no invasivo esté resuelto. Significa que hay una vía viable y medible para adaptar arquitecturas de contexto largo de MEG a EEG.

Y con esa lectura paso a cerrar con las conclusiones principales.

## 17. Conclusiones - 13:45-14:45

Como conclusión, en este TFM se ha adaptado una arquitectura inspirada en MEG-XL a EEG heterogéneo, se ha entrenado con datasets de lectura y escucha, y se ha evaluado en recuperación contextual de palabras sobre ds004408.

El resultado central es que el mejor modelo alcanza un 22,32 % de Top-10 balanceada, que son 5,58 veces el azar. Pero, para mí, el mensaje más importante no es solo el número final. El mensaje más importante es que sin preentrenamiento el modelo queda en azar, mientras que con preentrenamiento EEG aparece el salto principal. La transferencia desde MEG-XL suma, pero sobre una base EEG ya preentrenada.

Como trabajo futuro, quedan varias líneas claras: repetir con más semillas y particiones, estudiar mejor bandas y tokenizadores, ampliar la comparación a más datasets y, finalmente, explorar registros propios con casco EEG para controlar mejor el protocolo experimental.

La conclusión final sería: una arquitectura de contexto largo pensada inicialmente para MEG puede adaptarse a EEG y obtener resultados competitivos en una tarea controlada de clasificación contextual de palabras.

## 18. Gracias - 14:45-15:00

Con esto termino la presentación. Muchas gracias por vuestra atención, y quedo abierto a preguntas.

# Preguntas probables del tribunal

## ¿Por qué no hacer directamente generación de texto?

Porque en este TFM quería una tarea controlada y medible. La generación libre exige muchos más datos, una evaluación más compleja y protocolos diferentes. Aquí el objetivo es comprobar si la transferencia MEG-EEG ayuda en recuperación contextual de palabras.

## ¿El casco EEG puede leer pensamientos?

No. El casco registra potenciales eléctricos débiles en el cuero cabelludo, mezclados con artefactos. El modelo aprende asociaciones estadísticas entre señal EEG y palabras dentro de una tarea experimental concreta.

## ¿Cuál es la contribución principal?

La adaptación de una arquitectura de contexto largo inspirada en MEG-XL a EEG, junto con una evaluación que separa el efecto del preentrenamiento EEG, la inicialización desde MEG-XL y el embedding de sensor.

## ¿Qué resultado es más importante?

El salto de 4,05 % sin preentrenamiento a 19,95 % con preentrenamiento EEG. Ese es el efecto dominante. La transferencia desde MEG-XL aporta una mejora adicional, pero más moderada.

## ¿Por qué el mejor modelo reutiliza el embedding MEG si la señal final es EEG?

Mi lectura es que el embedding MEG no debe interpretarse literalmente como “esto sigue siendo MEG”, sino como una parametrización aprendida que funciona mejor en esta ejecución. Puede estar actuando como una inicialización útil para el tipo de representación que espera el checkpoint. Aun así, harían falta más semillas para asegurar que esta diferencia es estable.

## ¿Qué falta para que el resultado sea más sólido?

Más semillas, más particiones, más sujetos, comparación bajo protocolos idénticos y análisis específicos de bandas, tokenización y generalización entre sesiones.
