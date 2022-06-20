# Задание по кредитному скорингу
[Ссылка на презентацию](https://docs.google.com/presentation/d/1_bDAtTeu9eetR_yv-Hev1060xILNRwtp3KejIz_PBAs/edit#slide=id.p)

## Строение репозитория

- configs - конфигурации пайплайнов
- docs - вспомогательные документы
- models - обученные модели
- eda.ipynb - разведовательный анализ данных
- notebooks - ноутбуки с решением, которые могут быть запущены в Colab
  - colab_cat_boost.ipynb - обучение модели CatBoost
  - colab_log_reg.ipynb - обучение модели Logistic Regression
  - colab_el_var.ipynb - расчет expected и unexpected losses
- pipeline.py - всмоготательные классы и методы для настройки пайплайном
- utils.py - методы для загрузки и обработки данных
- woe.py - реализация woe для анализа
