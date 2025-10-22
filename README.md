Si las redes anteriores tenían una estructura comparable a Notredame, ahora se agrega una nueva capa.

https://youtu.be/iynoMdzFmpc?si=Nat_KXNrJErmxXkE

Forward separado por eteapas.

Antes tengo que decir que en este codigo se deja mucho a las librerías. Cosa que evite en los 3 codigos anteriores, usando librerías que no me dieran la estructura relativa a la red.

Pero habiendo aprendido MLP, CNN, RNN (LTSM incluido) me puedo permitir usar Torch, (una librería que automatiza varios procesos y nos ofrece estructuras incluidas).

Lo que puede jugar a favor o en contra, depende de como sea el programador. Yo por mi parte acostumbro a mirar mas el codigo por mi mismo y evitarlas. Al menos en lo que es aprender.

Por lo que, aqui en este resumen, no solo explicaré logica y estructura, sino tambien los puntos que importan a lo que es el uso de la librería.

Por mi parte, jamas había usado Torch. Se me hace que es un arma de doble filo. Oculta demaciado codigo a mi gusto. Y no imagino lo laborioso que puede volverse si llega a darnos algun problema.
Mirando un poco su codigo fuente, a mi gusto, no esta bien documentado.
Pero eso no quita que sin Torch, tendriamos que escribir mucho mas codigo, que en parte va aprendimos con RNN.

De nuevo, como en RNN, es dificil desarrollar un proyecto, que involucra una tecnología pensada para procesar cantidades inmensas de datos, que refleje buenamente el poder de lo que intentamos aprender.
Esta Seq2Seq tiene como proposito predecir una secuencia de numeros aprendida previamente.
De modo que si le dieramamos un numero, aprendiera que secuencia le sigue.
