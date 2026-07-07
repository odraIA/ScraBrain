# Guion de defensa - 15 minutos

Presentación: `presentacion_tfm/main.tex`.

La idea del guion es que cada diapositiva tenga muy poco texto y que el contenido fuerte lo pongas tú oralmente. El tono propuesto es académico, pero con una entrada divulgativa para que el tribunal entienda rápidamente por qué el problema es difícil.

## 0. Portada - 0:00-0:20

**Qué decir**

Buenas, soy Ricardo Díaz Peris y voy a presentar mi TFM, titulado *Transferencia de modelos MEG de contexto largo a EEG para clasificación contextual de palabras*. La idea central del trabajo es estudiar si un modelo aprendido con MEG, una señal no invasiva pero de mayor calidad, puede servir como punto de partida para trabajar con EEG, que es más ruidoso pero mucho más cercano a un uso real y portable.

## 1. La pregunta del trabajo - 0:20-1:05

**Qué decir**

La pregunta que resume el trabajo es esta: ¿puede un modelo aprendido con MEG ayudar a decodificar palabras desde EEG? MEG y EEG no son equivalentes, pero ambas observan actividad cerebral relacionada con el procesamiento del lenguaje. MEG ofrece mejores condiciones de señal, mientras que EEG es mucho más accesible. Mi trabajo intenta construir un puente entre ambas modalidades.

La métrica principal del resultado final es Top-10 balanceada con 250 palabras. El azar es un 4 %, y el mejor modelo alcanza un 22,32 %. Esto no significa que el problema de brain-to-text esté resuelto, pero sí que la adaptación queda claramente por encima del azar.

## 2. Por qué importa: comunicación sin cirugía - 1:05-1:55

**Qué decir**

El contexto general son las interfaces cerebro-computador aplicadas al lenguaje. A largo plazo, estas tecnologías podrían ayudar a personas que conservan la capacidad cognitiva de comunicarse pero no pueden articular habla. Los resultados más espectaculares suelen venir de registros invasivos, con electrodos implantados. Eso tiene mucho valor clínico, pero también riesgos y limitaciones prácticas.

Por eso es interesante explorar señales no invasivas. Este TFM no aborda todavía un sistema clínico completo, sino una tarea controlada: clasificación contextual de palabras alineadas temporalmente con la señal cerebral.

## 3. Qué capta realmente un casco de EEG - 1:55-2:55

**Qué decir**

Aquí puedes enseñar el casco si decides llevarlo. Yo lo usaría como recurso divulgativo, no como demostración técnica arriesgada. La explicación debe ser muy clara: el casco no lee pensamientos. Registra diferencias de potencial en el cuero cabelludo, del orden de microvoltios, que proceden de actividad neuronal sincronizada, pero mezcladas con mucho ruido.

Ese ruido incluye parpadeos, movimiento ocular, tensión muscular, mala impedancia de los electrodos y ruido eléctrico. Por tanto, el reto del modelo no es traducir pensamientos directamente, sino encontrar patrones estadísticos muy débiles en una señal muy contaminada.

**Consejo de defensa**

Si llevas el casco, úsalo durante 45-60 segundos. Enséñalo, señala electrodos y vuelve rápido al pipeline. Evita depender de una demo en directo que pueda fallar. Si quieres enseñar señal, mejor llevar una captura o vídeo ya preparado con señal cruda y un parpadeo marcado.

## 4. EEG y MEG no ven el cerebro igual - 2:55-3:45

**Qué decir**

EEG y MEG son no invasivas, pero miden magnitudes físicas distintas. EEG mide potenciales eléctricos sobre el cuero cabelludo; MEG mide campos magnéticos alrededor de la cabeza. EEG es barato y portable, pero tiene peor relación señal-ruido y peor resolución espacial. MEG tiene más calidad espacial y menos distorsión por tejidos, pero requiere equipamiento costoso y no portable.

Esto es importante porque mi hipótesis no es que EEG y MEG sean lo mismo. La hipótesis es más prudente: comprobar si un modelo de contexto largo entrenado con MEG puede aportar una inicialización útil cuando se adapta a EEG.

## 5. La hipótesis: aprender con contexto largo - 3:45-4:35

**Qué decir**

El lenguaje no ocurre en instantes aislados. Las palabras forman frases, las frases forman un contexto, y la señal cerebral también tiene dependencias temporales. MEG-XL parte precisamente de esta idea: en lugar de mirar ventanas muy cortas, modela fragmentos largos de señal.

En este trabajo mantengo esa idea de contexto largo y la llevo a EEG. Cada ejemplo de preentrenamiento contiene 150 segundos de señal, que posteriormente se dividen en bloques de tres segundos.

## 6. De MEG-XL a EEG-XL - 4:35-5:25

**Qué decir**

El flujo general es este. Partimos de MEG-XL como arquitectura y, en algunas condiciones, como checkpoint. Después construimos un adaptador EEG para poder representar montajes con distinto número de canales y posiciones de electrodos. A continuación se hace un preentrenamiento autosupervisado con EEG: primero lectura y después escucha. Finalmente, se ajusta el modelo en ds004408 para la tarea de clasificación contextual de palabras.

El objetivo no es cambiar todo el modelo, sino conservar lo que hace valioso a MEG-XL y adaptar la entrada a las características del EEG.

## 7. Datos: empezar por lectura, acabar en escucha - 5:25-6:20

**Qué decir**

La disponibilidad de datos EEG de lenguaje es una limitación importante. Por eso combino conjuntos de lectura y escucha. En lectura utilizo ZuCo 2.0 y Nieuwland; en escucha utilizo SparrKULee y OpenNeuro ds007808. Todos pasan por una interfaz común: canales EEG, posiciones de sensores, máscaras y ventanas de 150 segundos.

El conjunto ds004408 se reserva para el fine-tuning y la evaluación final. Esta separación es importante porque evita usar en el preentrenamiento el mismo conjunto sobre el que luego voy a medir la clasificación de palabras.

## 8. Preprocesamiento - 6:20-7:10

**Qué decir**

Como los datasets tienen frecuencias, montajes y escalas diferentes, el preprocesamiento es una parte central del trabajo. Primero se filtra la señal, después se remuestrea a 50 Hz, se normaliza y se extraen ventanas completas de 150 segundos. Además, se construye una representación común con posiciones de sensores y máscaras para indicar qué canales existen realmente en cada montaje.

Aunque el filtrado inicial llega hasta 40 Hz, al remuestrear a 50 Hz la frecuencia de Nyquist final limita la banda efectiva aproximadamente a 25 Hz. Esta decisión reduce mucho el coste computacional y mantiene compatibilidad con MEG-XL.

## 9. Arquitectura - 7:10-8:05

**Qué decir**

La arquitectura conserva los componentes principales de MEG-XL. BioCodec convierte cada canal en tokens discretos mediante cuantización vectorial residual. Después esos tokens se combinan con la información espacial del sensor, el tipo de sensor y la máscara de canales. El Transformer criss-cross modela relaciones temporales y espaciales sin tener que aplicar atención completa sobre todos los pares canal-tiempo.

Lo que cambia para EEG es sobre todo la representación de entrada: posiciones de electrodos, tipo de sensor EEG y máscaras para montajes heterogéneos.

## 10. Preentrenamiento autosupervisado - 8:05-8:55

**Qué decir**

Durante el preentrenamiento el modelo no ve palabras ni transcripciones. Se ocultan bloques de la señal y el modelo tiene que reconstruir los códigos de BioCodec. En cada ventana hay 50 bloques de tres segundos y se enmascaran 20.

Esto obliga al modelo a aprender regularidades temporales y espaciales del EEG sin depender de etiquetas lingüísticas. La hipótesis es que esa representación previa facilita después el ajuste supervisado sobre palabras.

## 11. Ajuste supervisado - 8:55-9:45

**Qué decir**

La parte supervisada utiliza ds004408. Para cada palabra se extrae una ventana de tres segundos alrededor de su inicio. Se agrupan 50 palabras, lo que vuelve a producir una entrada de 150 segundos. El modelo genera una representación cerebral y una cabeza de proyección la lleva al espacio de embeddings de T5.

La evaluación se formula como recuperación: se compara la predicción con embeddings de un vocabulario de 50 o 250 palabras y se comprueba si la palabra correcta está entre las 10 más cercanas.

## 12. Diseño experimental - 9:45-10:40

**Qué decir**

El diseño compara cuatro condiciones para separar efectos. La primera es un control sin preentrenamiento. La segunda preentrena EEG desde cero. La tercera parte de MEG-XL, pero usa un embedding específico de EEG. La cuarta también parte de MEG-XL, pero reutiliza directamente el embedding aprendido para magnetómetros.

Con esto puedo distinguir tres cosas: cuánto aporta el preentrenamiento EEG, cuánto aporta la transferencia desde MEG-XL y qué ocurre con la representación del tipo de sensor.

## 13. Resultado principal - 10:40-11:55

**Qué decir**

Este es el resultado central. El control sin preentrenamiento se queda prácticamente en azar: 4,05 % frente al 4 % esperado. El modelo preentrenado con EEG desde cero sube a 19,95 %. Esto muestra que el preentrenamiento autosupervisado es la parte más importante.

La transferencia desde MEG-XL añade una mejora más moderada. Con embedding EEG se obtiene 20,56 %, y reutilizando el embedding MEG se alcanza el mejor resultado: 22,32 %. En esta ejecución, la reutilización del embedding de magnetómetro funciona ligeramente mejor.

## 14. Qué nos dicen las curvas - 11:55-12:45

**Qué decir**

Las curvas de validación muestran otro punto importante: los mejores checkpoints aparecen muy pronto, entre las primeras épocas. Continuar el ajuste no mejora la validación e incluso puede deteriorarla. Esto es coherente con la naturaleza del EEG: hay mucha variabilidad y es fácil que un modelo de alta capacidad sobreajuste.

Por eso no solo importa el valor final, sino también la dinámica de entrenamiento. El preentrenamiento parece proporcionar una representación útil rápidamente, pero el fine-tuning debe controlarse con cuidado.

## 15. Lectura final de los resultados - 12:45-13:35

**Qué decir**

La lectura final es triple. Primero, la adaptación técnica de MEG-XL a EEG es viable. Segundo, el preentrenamiento EEG es la fuente principal de mejora. Tercero, la transferencia desde MEG-XL parece aportar, pero hay que confirmarlo con más semillas y protocolos más alineados con trabajos previos.

También hay que ser prudente: no es un sistema brain-to-text abierto, no genera frases libres y no se debe vender como lectura de pensamiento. Es una tarea controlada de recuperación de palabras que sirve como paso intermedio.

## 16. Casco EEG en la defensa - 13:35-14:10

**Qué decir**

Yo incluiría el casco porque ayuda a que el tribunal vea físicamente qué tipo de señal estamos intentando usar. La clave es integrarlo en el discurso: enseñar los electrodos, explicar que la señal es débil y ruidosa, y conectarlo con las decisiones del trabajo: filtrado, normalización, máscaras, preentrenamiento y cuidado con el sobreajuste.

No lo usaría como demo espectacular. Lo usaría como objeto explicativo, porque refuerza la parte divulgativa sin quitar seriedad.

## 17. Conclusión - 14:10-14:45

**Qué decir**

Como conclusión, la transferencia MEG-EEG no resuelve el problema completo, pero abre una vía útil. El mejor modelo alcanza 5,58 veces el azar en la métrica principal. La mayor mejora viene del preentrenamiento EEG y la transferencia desde MEG-XL añade una ganancia adicional.

El trabajo futuro pasa por repetir con más semillas, comparar tokenizadores y bandas de frecuencia, ampliar datasets y, especialmente, adquirir señales propias con cascos EEG para acercar el sistema a condiciones reales.

## 18. Gracias - 14:45-15:00

**Qué decir**

Muchas gracias. Quedo abierto a preguntas.

## Preguntas previsibles y respuestas cortas

### ¿Esto decodifica pensamiento?

No. El sistema no lee pensamientos ni genera texto libre. Evalúa una tarea controlada de recuperación de palabras alineadas temporalmente con estímulos conocidos.

### ¿Por qué usar EEG si MEG tiene mejor señal?

Porque EEG es más portable y viable en escenarios reales. MEG es útil como fuente de aprendizaje o como modalidad de alta calidad, pero no como dispositivo cotidiano.

### ¿Por qué el resultado de 22,32 % es relevante?

Porque la métrica principal tiene un azar del 4 %. El modelo queda claramente por encima del azar en una recuperación Top-10 balanceada sobre 250 palabras.

### ¿Qué aporta más: EEG o MEG-XL?

En esta ejecución, el salto dominante lo aporta el preentrenamiento autosupervisado con EEG. La transferencia desde MEG-XL añade una mejora adicional menor.

### ¿Qué haría después?

Repetir con más semillas, estudiar más bandas y tokenizadores, comparar con protocolos exactamente iguales a trabajos previos y crear un dataset propio de escucha continua con casco EEG.
