#  Aprendiendo-RL

Este repositorio documenta mi proceso de aprendizaje en **Reinforcement Learning**, abordando dos enfoques complementarios:

-  **Objetivo 2:** Uso de librerías de alto nivel como `stable-baselines3` para entrenar agentes rápidamente.
-  **Objetivo 3:** Implementación manual del algoritmo DQN en PyTorch, desde cero, con control total.

---

##  Estructura general del proyecto

```
Aprendiendo-RL/
├── Objetivo 2 LunarLander/
│   ├── models/              # Modelos .zip de SB3
│   ├── logs/                # TensorBoard
│   ├── outputs/             # Gráficas o evaluaciones
│   └── tests/               # Scripts de entrenamiento y evaluación
│
├── Objetivo 3 DQN-desde0/
│   ├── dqn_agent.py         # Clase DQN con red, buffer, entrenamiento
│   ├── train_lunarlander_dqn.py  # Entrenamiento principal
│   ├── eval_lunarlander_dqn.py   # Evaluación visual
│   ├── train_dqn.py         # DQN en CartPole
│   ├── eval_dqn.py          # Evaluación CartPole
│   ├── utils.py             # Gráficas
│   ├── outputs/             # Modelos .pth y gráficos .png
│
├── .gitignore               # Archivos a excluir del repo
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

---

##  Objetivo 2 – LunarLander con `stable-baselines3`

En esta carpeta entreno agentes en `LunarLander-v2` usando `stable-baselines3`.  
Incluye:

- Entrenamiento con DQN preconfigurado
- Uso de `VecMonitor` para logging
- Visualización con TensorBoard
- Pruebas con distintos hiperparámetros
- Evaluación con render en `eval_lunarlander.py`

### Ejecutar un entrenamiento:
```bash
cd "Objetivo 2 LunarLander/tests"
python train_lunarlander.py
```

---

##  Objetivo 3 – DQN desde cero en PyTorch

Aquí desarrollo una implementación completa de **Deep Q-Learning** sin librerías externas de RL, con énfasis en:

- Arquitectura MLP configurable
- Replay buffer manual
- Target network con `update_target()`
- Política epsilon-greedy (con decaimiento lineal controlado)
- Entrenamiento en `LunarLander-v2` y `CartPole-v1`

### Entrenar LunarLander:
```bash
cd "Objetivo 3 DQN-desde0"
python train_lunarlander_dqn.py
```

### Evaluar visualmente:
```bash
python eval_lunarlander_dqn.py
```

---

##  Requisitos

Instala todas las dependencias con:

```bash
pip install -r requirements.txt
```

Contenido recomendado en `requirements.txt`:

```
torch
gymnasium[box2d]
matplotlib
```

---

##  Próximos pasos

- [ ] Añadir Double DQN
- [ ] Implementar Prioritized Experience Replay
- [ ] Soporte para `LunarLander-v3`
- [ ] Migración a entornos más complejos (BipedalWalker, CarRacing)
- [ ] Documentación con diagramas

---

##  Autor

**Izanvz**  
[github.com/Izanvz](https://github.com/Izanvz)

izanvillarejo2002@gmail.com

