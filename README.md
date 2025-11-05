# 🚀 Home Credit Risk Analysis

Bienvenido al proyecto **Home Credit Risk Analysis**, desarrollado como parte del curso de **Machine Learning**.  
Este repositorio contiene el código fuente, los notebooks y la documentación en formato de **Jupyter Book**, desplegada en GitHub Pages.  

👉 **Acceso directo al libro**  
[![Ver Jupyter Book](https://img.shields.io/badge/📖%20Ver%20Jupyter%20Book-blueviolet)](https://nicoplayz58.github.io/Home-Credit-Risky-Analysis/)

---

## 🌟 Descripción

El acceso al crédito es un pilar fundamental del desarrollo económico, pero evaluar la solvencia de los solicitantes sigue siendo un desafío crítico.  
Este proyecto aborda la competencia **[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)**, la cual busca predecir la probabilidad de que un cliente incumpla el pago de su préstamo.  

A través del uso de **técnicas de Machine Learning supervisado**, se desarrollan modelos que permiten identificar a los clientes con mayor riesgo, mejorando la eficiencia en la toma de decisiones crediticias.  

---

## 🎯 Objetivos del Proyecto

- Analizar y comprender la estructura del dataset.  
- Identificar y tratar valores faltantes, outliers y correlaciones.  
- Explorar las variables más relevantes para la predicción del riesgo.  
- Entrenar y evaluar distintos modelos de clasificación.  
- Comparar resultados mediante métricas de desempeño.  

---

## 🧩 Metodología

El flujo de trabajo se divide en tres etapas principales:

1. **Contexto del Problema**  
   Explicación del dataset, propósito del análisis y justificación metodológica.  

2. **Análisis Exploratorio de Datos (EDA)**  
   - Distribuciones y estadísticas descriptivas.  
   - Correlaciones y análisis de variables clave.  
   - Visualización de relaciones entre características y variable objetivo.  

3. **Modelos Predictivos**  
   - Preprocesamiento y selección de características.  
   - Entrenamiento de modelos como **Logistic Regression**, **Random Forest**, **XGBoost** y **SVM**.  
   - Evaluación de métricas: *Accuracy*, *ROC-AUC*, *Precision*, *Recall* y *F1-score*.  

---

## 📂 Estructura del Repositorio

```bash
homecredit-book/
├── contexto.md                # Portada y descripción del problema
├── EDA_Home_Credit_V2.ipynb   # Análisis exploratorio de datos
├── Modelos_Home_Credit.ipynb  # Entrenamiento y evaluación de modelos
├── _config.yml                # Configuración del Jupyter Book
├── _toc.yml                   # Tabla de contenidos del libro
├── _build/                    # Carpeta generada con el HTML del sitio
└── README.md                  # Este archivo
