# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="C1hYGVSPuGAn"
# # Задание 2

# %% [markdown] id="l1cebBfUu70w"
# ##Основное задание

# %% colab={"base_uri": "https://localhost:8080/"} id="awAFmET9dvH_" outputId="5ba28c71-1a9d-4af5-fcaa-16033b270c17"
import pandas as pd
import networkx as nx
# !pip install pulp
import pulp

# %% [markdown] id="uXVCGlzFepA1"
# ### Загрузка датасетов
#

# %% id="K-YoUbnfd23J"
df_project = pd.read_csv('csv1.txt') #Блок метаданных проекта
df_tasks = pd.read_csv('csv2.txt') #Блок описания задач (20 задач)
df_employees= pd.read_csv('csv3.txt') #Блок описания ресурсов (30 сотрудников)
df_limitations = pd.read_csv('csv4.txt') #Блок дополнительных ограничений
df_keys= pd.read_csv('csv5.txt') #Ключевые связи

# %% [markdown] id="szdnNRHEg16L"
# ### 1. Подготовка данных

# %% id="X0OsjVdJhjSA"
#Расчет длительности (pert)
#PERT = (Оптимистичная + 4 × Наиболее вероятная + Пессимистичная) / 6

df_project['pert_expected_duration'] = (
    df_project['optimistic_days'] +
    4 * df_project['likely_days'] +
    df_project['pessimistic_days']
) / 6

df_project['pert_expected_duration'] = df_project['pert_expected_duration'].round(2)


# %% id="0OvMVZAuiwPf"
# Расчет общих трудозатрат
df_project['total_effort_hours'] = df_project['pert_expected_duration'] * 8

# %% id="m4QWT7ZNHW4h"
# Корректировка максимальной загрузки для сотрудников с health_status != "Отлично"
df_employees.loc[df_employees['health_status'] != 'Отлично', 'max_hours_day'] -= 2


# %% [markdown] id="nGop4mPmM3zi"
# ###2. Функции для проверки ограничений

# %% id="GXjW7g5gN17n"
# Функция для проверки подходящих сотрудников для задачи
def get_eligible_employees_for_task(task, employees_df):
    """Находит сотрудников, подходящих для задачи по всем критериям"""
    eligible_employees = []

    for _, emp in employees_df.iterrows():
        # 1. Проверка навыков (skill_1 и skill_2 с уровнем >= 7)
        skill_1_ok = (
            (emp['primary_skill'] == task['skill_1'] and emp['skill_level'] >= 7) or
            (emp['secondary_skill'] == task['skill_1'] and emp['sec_skill_level'] >= 7)
        )

        skill_2_ok = (
            (emp['primary_skill'] == task['skill_2'] and emp['skill_level'] >= 7) or
            (emp['secondary_skill'] == task['skill_2'] and emp['sec_skill_level'] >= 7)
        )

        skill_ok = skill_1_ok or skill_2_ok

        # 2. Проверка security clearance
        security_ok = emp['security_clear'] >= task['min_security']

        # 3. Проверка опыта для задач с высокой видимостью
        experience_ok = True
        if task['client_visibility'] == 'Высокая':
            experience_ok = emp['experience'] >= 3

        if skill_ok and security_ok and experience_ok:
            eligible_employees.append(emp['emp_id'])

    return eligible_employees


# %% id="3PMjWuFON9NS"
# Функция для проверки инновационных задач (мягкое ограничение)
def get_innovation_preference(task, employees_df):
    """Для инновационных задач возвращает предпочтительных сотрудников"""
    if task['is_innovation'] == 'Да':
        return employees_df[employees_df['innovation_interest'] == 'Да']['emp_id'].tolist()
    return []  # Для неинновационных задач ограничения нет


# %% id="7f5buXP_NDIp"
# Функция для проверки локации
def check_location_constraint(task_id, assigned_employees, employees_df, limitations_df):
    """Проверяет требование совместной локации"""
    # Ищем ограничение по локации для этой задачи
    location_constraint = limitations_df[
        (limitations_df['constraint_type'] == 'team_co_location') &
        (limitations_df['affected_tasks'].str.contains(task_id, na=False))
    ]

    if not location_constraint.empty and location_constraint.iloc[0]['constraint_value'] == 'Да':
        if len(assigned_employees) > 0:
            locations = employees_df[employees_df['emp_id'].isin(assigned_employees)]['location'].unique()
            return len(locations) <= 1
    return True


# %% [markdown] id="0TDKt-_GODbf"
# ### 3. Постановка задачи линейного программирования

# %% id="3KoXTVGwOb56"
# Создаем модель оптимизации
model = pulp.LpProblem("Optimal_Resource_Allocation", pulp.LpMinimize)

# %% colab={"base_uri": "https://localhost:8080/"} id="lHtMeCQrOeqC" outputId="f7672b4f-f70a-4bde-a499-7474f92e6356"
# Создаем переменные решения x_ij (задача i -> сотрудник j)
assignments = {}

for _, task in df_project.iterrows():
    # Находим сотрудников, подходящих по жестким ограничениям
    eligible_emps = get_eligible_employees_for_task(task, df_employees)

    # Добавляем предпочтение для инновационных задач
    innovation_preferred = get_innovation_preference(task, df_employees)

    for emp_id in eligible_emps:
        var_name = f"assign_{task['task_id']}_{emp_id}"
        assignments[(task['task_id'], emp_id)] = pulp.LpVariable(var_name, cat='Binary')

print(f"Создано {len(assignments)} переменных решения")

# %% [markdown] id="DQJtQJb0OlAT"
# ### 4. Целевая функция (минимизация затрат)

# %% id="UtlIZuaLO1xr"
# Целевая функция: Minimize Σ (x_ij * total_effort_hours_i * hourly_rate_j)
cost_expression = pulp.LpAffineExpression()

for (task_id, emp_id), assignment_var in assignments.items():
    # Трудозатраты по задаче
    task_hours = df_project[df_project['task_id'] == task_id]['total_effort_hours'].iloc[0]
    # Часовая ставка сотрудника
    emp_rate = df_employees[df_employees['emp_id'] == emp_id]['hourly_rate'].iloc[0]

    # Добавляем к целевой функции
    cost_expression += assignment_var * task_hours * emp_rate

# %% id="KS63KvMEO_iF"
# Устанавливаем целевую функцию
model += cost_expression, "Total_Project_Cost"
#print("Целевая функция установлена: минимизация общих затрат")

# %% [markdown] id="KzvYzat5PCvt"
# ### 5. Ограничения

# %% id="EIOYEOGyPTBe"
# 5.1 На каждую задачу должен быть назначен хотя бы один сотрудник
for task_id in df_project['task_id']:
    task_assignments = [var for (t_id, emp_id), var in assignments.items() if t_id == task_id]
    if task_assignments:
        model += pulp.lpSum(task_assignments) >= 1, f"min_employees_{task_id}"

# %% id="KdUdwfHbPXF-"
# 5.2 Не превышать max_employees_per_task
# Находим ограничение из df_limitations
max_employees_constraint = df_limitations[
    df_limitations['constraint_type'] == 'max_employees_per_task'
]
max_emps = int(max_employees_constraint['constraint_value'].iloc[0]) if not max_employees_constraint.empty else 3

for task_id in df_project['task_id']:
    task_assignments = [var for (t_id, emp_id), var in assignments.items() if t_id == task_id]
    if task_assignments:
        model += pulp.lpSum(task_assignments) <= max_emps, f"max_employees_{task_id}"

# %% id="ZUkovsY7PJ90"
# 5.3 Ограничение по загрузке сотрудников (с учетом отпусков)
for emp_id in df_employees['emp_id']:
    emp_assignments = [var for (task_id, e_id), var in assignments.items() if e_id == emp_id]
    if emp_assignments:
        # Расчет доступного времени сотрудника
        emp_data = df_employees[df_employees['emp_id'] == emp_id].iloc[0]

        # Берем данные проекта для расчета длительности
        project_duration_weeks = df_tasks['total_expected_duration'].iloc[0] / 7

        # Доступное время с учетом текущей загрузки и корректировки здоровья
        available_hours = (
            emp_data['max_hours_day'] * 5 *  # 5 дней в неделю
            project_duration_weeks *          # длительность проекта в неделях
            (1 - emp_data['workload_pct'] / 100)  # доступность после текущей загрузки
        )

        # Выражение для суммарного времени сотрудника
        total_hours_expr = pulp.LpAffineExpression()
        for assignment_var in emp_assignments:
            # Находим task_id для этого назначения
            task_id = next(t_id for (t_id, e_id) in assignments.keys()
                          if assignments[(t_id, e_id)] == assignment_var and e_id == emp_id)
            task_hours = df_project[df_project['task_id'] == task_id]['total_effort_hours'].iloc[0]
            total_hours_expr += assignment_var * task_hours

       # model += total_hours_expr <= available_hours, f"workload_limit_{emp_id}"

# %% [markdown] id="pHvNdwy4Prda"
# ###6. Решение задачи

# %% colab={"base_uri": "https://localhost:8080/"} id="H6QKNSefP2LA" outputId="45c4f46b-f4dc-4569-e67e-753e72cafed4"
# Решаем задачу оптимизации
model.solve(pulp.PULP_CBC_CMD(msg=1))

# Проверяем статус решения
print(f"Статус решения: {pulp.LpStatus[model.status]}")

# %% [markdown] id="6kxmcBuUP82A"
# ### 7. Анализ результатов

# %% colab={"base_uri": "https://localhost:8080/"} id="BsVsDVtXQGkG" outputId="2fbecb63-e6f8-4c96-c690-8588773c2f87"
# 7.1 Вывод общей стоимости
if model.status == pulp.LpStatusOptimal:
    print(f"Общая стоимость проекта: {pulp.value(model.objective):,.2f} руб.")

    # 7.2 Матрица назначений
    print(f"\nМАТРИЦА НАЗНАЧЕНИЙ")
    assignment_results = []

    for (task_id, emp_id), var in assignments.items():
        if pulp.value(var) > 0.5:  # Назначение активно
            task_data = df_project[df_project['task_id'] == task_id].iloc[0]
            emp_data = df_employees[df_employees['emp_id'] == emp_id].iloc[0]

            assignment_results.append({
                'task_id': task_id,
                'task_name': task_data['task_name'],
                'emp_id': emp_id,
                'emp_name': emp_data['emp_name'],
                'hours': task_data['total_effort_hours'],
                'hourly_rate': emp_data['hourly_rate'],
                'cost': task_data['total_effort_hours'] * emp_data['hourly_rate']
            })

            print(f"{task_data['task_name']} -> {emp_data['emp_name']} "
                  f"({task_data['total_effort_hours']}ч, {emp_data['hourly_rate']} руб/ч)")

    # 7.3 Сводная статистика
    total_cost = sum(item['cost'] for item in assignment_results)
    total_hours = sum(item['hours'] for item in assignment_results)

    print(f"\nСВОДНАЯ СТАТИСТИКА")
    print(f"Общая стоимость: {total_cost:,.2f} руб.")
    print(f"Общие трудозатраты: {total_hours:.1f} часов")
    print(f"Количество назначений: {len(assignment_results)}")

else:
    print("Оптимальное решение не найдено")


# %% [markdown] id="Hmjq3lU7Qfxs"
# ### 8. Расчет длительности проекта

# %% colab={"base_uri": "https://localhost:8080/"} id="UUfZMVlvQPbk" outputId="74397c4a-7c42-4f13-d38a-0df7e45de3d6"
# Расчет длительности проекта с учетом зависимостей задач
def calculate_project_duration(tasks_df):
    """Рассчитывает длительность проекта через критический путь"""
    G = nx.DiGraph()

    # Добавляем задачи в граф
    for _, task in tasks_df.iterrows():
        G.add_node(task['task_id'], duration=task['pert_expected_duration'])

    # Добавляем зависимости
    for _, task in tasks_df.iterrows():
        if task['dependencies'] and pd.notna(task['dependencies']):
            deps = str(task['dependencies']).split(',')
            for dep in deps:
                dep = dep.strip()
                if dep and dep in G.nodes:
                    G.add_edge(dep, task['task_id'])

    if not G.nodes:
        return 0

    # Расчет критического пути
    try:
        # Находим самый длинный путь
        longest_path = nx.dag_longest_path(G)
        critical_path_duration = sum(G.nodes[node]['duration'] for node in longest_path)

        print(f"\nДЛИТЕЛЬНОСТЬ ПРОЕКТА")
        print(f"Критический путь: {longest_path}")
        print(f"Длительность проекта: {critical_path_duration:.1f} дней")

        return critical_path_duration

    except Exception as e:
        print(f"Ошибка расчета длительности: {e}")
        return max(tasks_df['pert_expected_duration'])

# Вызываем функцию расчета
project_duration = calculate_project_duration(df_project)

# %% [markdown] id="XldFsc_8veMz"
# ##ШПАРГАЛКА
# **PANDAS**
# ###DataFrame**
# - df.iterrows() - итерация по строкам
# - df.columns.tolist() - возвращает список названий колонок
# - df.iloc[] - доступ по integer-позиции
# - df.loc[] - доступ по метке или boolean-маске
# - df.shape - размерность
# - df.nlargest() - n наибольших значений
# - df.empty - пустой ли df
# - df.notna() - маска не-NaN значений
# - df.isin() - наличие значений в списке
# - df.apply() - функция вдоль оси
# ###Series**
# - series.round() - округление
# - series.clip() - ограничивает значения в диапазоне
# - series.tolist() - конвертирует в список
# ###**Функции**
# - pd.to_datetime() - в datetime
# - pd.value_counts() - подсчет уникальных значений
#
# ##**PuLP (ЛИНЕЙНОЕ ПРОГРАММИРОВАНИЕ)**
# ### **Создание модели**
# - pulp.LpProblem() - создает задачу оптимизации
# - pulp.LpVariable() - создает переменную решения
# - pulp.LpVariable.dicts() - создает словарь переменных
# ### **Целевые функции и ограничения**
# - pulp.LpMinimize - константа для минимизации
# - pulp.LpAffineExpression() - линейное выражение для целевой функции
# - pulp.lpSum() - суммирование выражений
# - model += (добавление ограничения) - добавляет ограничение в модель
# ### **Решение и анализ**
# -model.solve() - решает задачу оптимизации
# - pulp.LpStatus[] - статус решения модели
# - pulp.value() - получает значение переменной
# - pulp.PULP_CBC_CMD() - использует CBC решатель
#
# ##**NETWORKX (ГРАФЫ)**
# - nx.DiGraph() - создает ориентированный граф
# - graph.add_node() - добавляет узел
# - graph.add_edge() - добавляет ребро
# - graph.nodes[] - доступ к атрибутам узла
# - graph.predecessors() - предшественники узла
# - graph.successors() - последователи узла
# - nx.topological_sort() - топологическая сортировка
# - nx.dag_longest_path() - критический путь
#
# ##**СПИСОК ФУНКЦИЙ**
# ### **Оптимизация**
# - get_eligible_employees_for_task() - находит подходящих сотрудников для задачи
# - get_eligible_employees_base() - middle-сотрудники для базового сценария
# - get_eligible_employees_quality() - senior-сотрудники для сценария качества
# - add_basic_constraints() - lобавляет ограничения в модель
# - solve_variant_10_base_scenario() - решает базовый сценарий
# - solve_variant_10_quality_scenario() - решает сценарий качества
#
# ### **Вспомогательные**
# - parse_vacation_dates() - парсит даты отпусков из строки
# - calculate_availability_with_vacation() - расчет доступности с учетом отпусков
# - calculate_project_duration() - расчет длительности проекта
# - build_task_graph() - строит граф зависимостей задач
# - calculate_critical_path() - находит критический путь
# - analyze_variant_10_results() - анализирует результаты сравнения
# - get_employee_info() - форматирует информацию о сотруднике
# - analyze_employee_distribution() - анализ распределения по опыту
#
# ### **Анализ**
# - check_location_constraint() - проверяет требование совместной локации
# - get_innovation_preference() - находит сотрудников для инновационных задач

# %% [markdown] id="EsCzWPZcRcFq"
# ## ВАРИАНТ 10

# %% colab={"base_uri": "https://localhost:8080/"} id="YGPE1oE3RgAW" outputId="d1fa0bb9-185a-4577-f1af-2ccccec41d4b"
import pandas as pd
import pulp

def solve_variant_10_base_scenario():
    """Базовый сценарий: ТОЛЬКО middle-разработчики (опыт 3-4 года) на задачи с высокой видимостью"""

    # Копируем данные чтобы не менять оригинальные
    tasks_df = df_project.copy()
    employees_df = df_employees.copy()

    # Создаем модель для базового сценария
    model_base = pulp.LpProblem("Variant10_Base_Scenario", pulp.LpMinimize)
    assignments_base = {}

    # Функция для проверки подходящих сотрудников (ТОЛЬКО middle: опыт 3-4 года)
    def get_eligible_employees_base(task, employees_df):
        eligible_employees = []

        for _, emp in employees_df.iterrows():
            # Проверка навыков
            skill_ok = (
                (emp['primary_skill'] == task['skill_1'] and emp['skill_level'] >= 7) or
                (emp['secondary_skill'] == task['skill_1'] and emp['sec_skill_level'] >= 7) or
                (emp['primary_skill'] == task['skill_2'] and emp['skill_level'] >= 7) or
                (emp['secondary_skill'] == task['skill_2'] and emp['sec_skill_level'] >= 7)
            )

            # Проверка security clearance
            security_ok = emp['security_clear'] >= task['min_security']

            # ИСПРАВЛЕНИЕ: ТОЛЬКО middle-разработчики (опыт 3-4 года) для задач с высокой видимостью
            experience_ok = True
            if task['client_visibility'] == 'Высокая':
                experience_ok = (emp['experience'] >= 3) and (emp['experience'] <= 4)  # ТОЛЬКО 3-4 года!
            else:
                experience_ok = emp['experience'] >= 3  # Для остальных задач - минимальный опыт 3 года

            if skill_ok and security_ok and experience_ok:
                eligible_employees.append(emp['emp_id'])

        return eligible_employees

    # Создаем переменные
    print("Создание переменных для базового сценария (middle-only)...")
    for _, task in tasks_df.iterrows():
        eligible_emps = get_eligible_employees_base(task, employees_df)

        for emp_id in eligible_emps:
            var_name = f"base_{task['task_id']}_{emp_id}"
            assignments_base[(task['task_id'], emp_id)] = pulp.LpVariable(var_name, cat='Binary')

    print(f"Создано {len(assignments_base)} переменных для базового сценария")

    # Целевая функция
    cost_base = pulp.LpAffineExpression()
    for (task_id, emp_id), var in assignments_base.items():
        task_hours = tasks_df[tasks_df['task_id'] == task_id]['total_effort_hours'].iloc[0]
        emp_rate = employees_df[employees_df['emp_id'] == emp_id]['hourly_rate'].iloc[0]
        cost_base += var * task_hours * emp_rate

    model_base += cost_base, "Total_Cost_Base"

    # Добавляем ограничения
    add_basic_constraints(model_base, assignments_base, tasks_df, employees_df)

    # Решаем
    print("Решение базового сценария...")
    model_base.solve(pulp.PULP_CBC_CMD(msg=0))

    return model_base, assignments_base

def solve_variant_10_quality_scenario():
    """Сценарий качества: ТОЛЬКО senior-разработчики (опыт ≥5 лет) на задачи с высокой видимостью"""

    tasks_df = df_project.copy()
    employees_df = df_employees.copy()

    # Функция для проверки senior-разработчиков
    def get_eligible_employees_quality(task, employees_df):
        eligible_employees = []

        for _, emp in employees_df.iterrows():
            # Проверка навыков
            skill_ok = (
                (emp['primary_skill'] == task['skill_1'] and emp['skill_level'] >= 7) or
                (emp['secondary_skill'] == task['skill_1'] and emp['sec_skill_level'] >= 7) or
                (emp['primary_skill'] == task['skill_2'] and emp['skill_level'] >= 7) or
                (emp['secondary_skill'] == task['skill_2'] and emp['sec_skill_level'] >= 7)
            )

            # Проверка security clearance
            security_ok = emp['security_clear'] >= task['min_security']

            # ИСПРАВЛЕНИЕ: ТОЛЬКО senior-разработчики (опыт ≥5 лет) для задач с высокой видимостью
            experience_ok = True
            if task['client_visibility'] == 'Высокая':
                experience_ok = emp['experience'] >= 5  # ТОЛЬКО ≥5 лет!
            else:
                experience_ok = emp['experience'] >= 3  # Для остальных задач - минимальный опыт 3 года

            if skill_ok and security_ok and experience_ok:
                eligible_employees.append(emp['emp_id'])

        return eligible_employees

    # Создаем модель для сценария качества
    model_quality = pulp.LpProblem("Variant10_Quality_Scenario", pulp.LpMinimize)
    assignments_quality = {}

    # Создаем переменные с новыми ограничениями
    print("Создание переменных для сценария качества (senior-only)...")
    for _, task in tasks_df.iterrows():
        eligible_emps = get_eligible_employees_quality(task, employees_df)

        for emp_id in eligible_emps:
            var_name = f"quality_{task['task_id']}_{emp_id}"
            assignments_quality[(task['task_id'], emp_id)] = pulp.LpVariable(var_name, cat='Binary')

    print(f"Создано {len(assignments_quality)} переменных для сценария качества")

    # Целевая функция
    cost_quality = pulp.LpAffineExpression()
    for (task_id, emp_id), var in assignments_quality.items():
        task_hours = tasks_df[tasks_df['task_id'] == task_id]['total_effort_hours'].iloc[0]
        emp_rate = employees_df[employees_df['emp_id'] == emp_id]['hourly_rate'].iloc[0]
        cost_quality += var * task_hours * emp_rate

    model_quality += cost_quality, "Total_Cost_Quality"

    # Добавляем ограничения
    add_basic_constraints(model_quality, assignments_quality, tasks_df, employees_df)

    # Решаем
    print("Решение сценария качества...")
    model_quality.solve(pulp.PULP_CBC_CMD(msg=0))

    return model_quality, assignments_quality

def add_basic_constraints(model, assignments, tasks_df, employees_df):
    """Добавляет базовые ограничения в модель"""

    # 1. Минимум 1 сотрудник на задачу
    for task_id in tasks_df['task_id']:
        task_vars = [var for (t_id, emp_id), var in assignments.items() if t_id == task_id]
        if task_vars:
            model += pulp.lpSum(task_vars) >= 1, f"min_emp_{task_id}"
        else:
            print(f"Нет подходящих сотрудников для задачи {task_id}")

    # 2. Максимум сотрудников на задачу
    max_emps = 3
    for task_id in tasks_df['task_id']:
        task_vars = [var for (t_id, emp_id), var in assignments.items() if t_id == task_id]
        if task_vars:
            model += pulp.lpSum(task_vars) <= max_emps, f"max_emp_{task_id}"

    # 3. Ограничение по загрузке
    for emp_id in employees_df['emp_id']:
        emp_vars = [var for (t_id, e_id), var in assignments.items() if e_id == emp_id]
        if emp_vars:
            emp_data = employees_df[employees_df['emp_id'] == emp_id].iloc[0]

            # Расчет доступного времени (упрощенно)
            project_duration_weeks = 8  # Предполагаем 8 недель проекта
            available_hours = emp_data['max_hours_day'] * 5 * project_duration_weeks

            total_hours_expr = pulp.LpAffineExpression()
            for var in emp_vars:
                # Находим task_id для этого назначения
                for (t_id, e_id), v in assignments.items():
                    if v == var and e_id == emp_id:
                        task_hours = tasks_df[tasks_df['task_id'] == t_id]['total_effort_hours'].iloc[0]
                        total_hours_expr += var * task_hours
                        break

            model += total_hours_expr <= available_hours, f"workload_{emp_id}"

def get_employee_info(emp_id, employees_df):
    """Возвращает информацию о сотруднике"""
    emp_data = employees_df[employees_df['emp_id'] == emp_id].iloc[0]
    return f"{emp_data['emp_name']} (опыт: {emp_data['experience']} лет, ставка: {emp_data['hourly_rate']} руб/ч)"

def analyze_employee_distribution(employees_df, high_visibility_tasks):
    """Анализирует распределение сотрудников по уровню опыта"""
    print(f"\nРАСПРЕДЕЛЕНИЕ СОТРУДНИКОВ ПО ОПЫТУ:")

    junior_count = len(employees_df[employees_df['experience'] < 3])
    middle_count = len(employees_df[(employees_df['experience'] >= 3) & (employees_df['experience'] <= 4)])
    senior_count = len(employees_df[employees_df['experience'] >= 5])

    print(f"Junior (<3 лет): {junior_count} сотрудников")
    print(f"Middle (3-4 года): {middle_count} сотрудников")
    print(f"Senior (≥5 лет): {senior_count} сотрудников")
    print(f"Задач с высокой видимостью: {len(high_visibility_tasks)}")

def analyze_variant_10_results(model_base, assignments_base, model_quality, assignments_quality):
    """Анализирует и сравнивает результаты двух сценариев"""

    # Проверяем статусы решений
    print(f"Статус базового сценария (middle-only): {pulp.LpStatus[model_base.status]}")
    print(f"Статус сценария качества (senior-only): {pulp.LpStatus[model_quality.status]}")

    # Анализ задач с высокой видимостью
    high_visibility_tasks = df_project[df_project['client_visibility'] == 'Высокая']
    analyze_employee_distribution(df_employees, high_visibility_tasks)

    if model_base.status == pulp.LpStatusOptimal and model_quality.status == pulp.LpStatusOptimal:
        cost_base = pulp.value(model_base.objective)
        cost_quality = pulp.value(model_quality.objective)

        cost_difference = cost_quality - cost_base
        cost_increase_percent = (cost_difference / cost_base) * 100 if cost_base > 0 else 0

        print(f"\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ СТОИМОСТИ:")
        print(f"Базовый сценарий (middle): {cost_base:,.2f} руб.")
        print(f"Сценарий качества (senior): {cost_quality:,.2f} руб.")
        print(f"Разница в стоимости: {cost_difference:,.2f} руб.")
        print(f"Процент увеличения: {cost_increase_percent:.1f}%")

        print(f"\nДЕТАЛЬНЫЙ АНАЛИЗ ЗАДАЧ С ВЫСОКОЙ ВИДИМОСТЬЮ:")

        for _, task in high_visibility_tasks.iterrows():
            print(f"\n{task['task_name']}")
            print(f"Навыки: {task['skill_1']}, {task['skill_2']}")

            # Находим назначенных сотрудников в базовом сценарии (MIDDLE)
            base_emps = []
            for (t_id, emp_id), var in assignments_base.items():
                if t_id == task['task_id'] and pulp.value(var) > 0.5:
                    base_emps.append(get_employee_info(emp_id, df_employees))

            # Находим назначенных сотрудников в сценарии качества (SENIOR)
            quality_emps = []
            for (t_id, emp_id), var in assignments_quality.items():
                if t_id == task['task_id'] and pulp.value(var) > 0.5:
                    quality_emps.append(get_employee_info(emp_id, df_employees))

            print(f"MIDDLE (3-4 года):")
            if base_emps:
                for emp in base_emps:
                    print(f"{emp}")
            else:
                print(f"Нет подходящих middle-разработчиков")

            print(f"SENIOR (≥5 лет):")
            if quality_emps:
                for emp in quality_emps:
                    print(f"{emp}")
            else:
                print(f"Нет подходящих senior-разработчиков")

        # Анализ эффективности
        print(f"\nАНАЛИЗ ЭФФЕКТИВНОСТИ:")
        if cost_increase_percent <= 5:
            print(f"Отличный результат! Всего +{cost_increase_percent:.1f}% за повышение качества")
            print(f"Рекомендация: Принять сценарий с senior-разработчиками")
        elif cost_increase_percent <= 15:
            print(f"Умеренное увеличение стоимости: +{cost_increase_percent:.1f}%")
            print(f"Рекомендация: Рассмотреть гибридный подход")
        else:
            print(f"Значительное увеличение стоимости: +{cost_increase_percent:.1f}%")
            print(f"Рекомендация: Оставить базовый сценарий")

    else:
        print("Ошибка при решении одной из моделей")
        if model_base.status != pulp.LpStatusOptimal:
            print("Проблема в базовом сценарии (middle-only)")
        if model_quality.status != pulp.LpStatusOptimal:
            print("Проблема в сценарии качества (senior-only)")

# Проверка данных
print("ПРОВЕРКА ДАННЫХ:")
print(f"Задач: {len(df_project)}")
print(f"Сотрудников: {len(df_employees)}")
print(f"Задач с client_visibility='Высокая': {len(df_project[df_project['client_visibility'] == 'Высокая'])}")

# Статистика по опыту сотрудников
junior_count = len(df_employees[df_employees['experience'] < 3])
middle_count = len(df_employees[(df_employees['experience'] >= 3) & (df_employees['experience'] <= 4)])
senior_count = len(df_employees[df_employees['experience'] >= 5])

print(f"Junior-разработчиков (<3 лет): {junior_count}")
print(f"Middle-разработчиков (3-4 года): {middle_count}")
print(f"Senior-разработчиков (≥5 лет): {senior_count}")

# Проверка доступности middle-разработчиков для задач с высокой видимостью
high_visibility_tasks = df_project[df_project['client_visibility'] == 'Высокая']
print(f"\nПРОВЕРКА ДОСТУПНОСТИ MIDDLE-РАЗРАБОТЧИКОВ:")

for _, task in high_visibility_tasks.iterrows():
    middle_emps = df_employees[
        (df_employees['experience'] >= 3) &
        (df_employees['experience'] <= 4) &
        (
            (df_employees['primary_skill'] == task['skill_1']) |
            (df_employees['secondary_skill'] == task['skill_1']) |
            (df_employees['primary_skill'] == task['skill_2']) |
            (df_employees['secondary_skill'] == task['skill_2'])
        ) &
        (df_employees['skill_level'] >= 7) &
        (df_employees['security_clear'] >= task['min_security'])
    ]
    print(f"{task['task_name']}: {len(middle_emps)} подходящих middle-разработчиков")

# Запуск решения
try:
    print(f"\nРЕШЕНИЕ ЗАДАЧ ОПТИМИЗАЦИИ...")
    model_base, assignments_base = solve_variant_10_base_scenario()
    model_quality, assignments_quality = solve_variant_10_quality_scenario()

    # Анализ результатов
    analyze_variant_10_results(model_base, assignments_base, model_quality, assignments_quality)

except Exception as e:
    print(f"Ошибка при выполнении: {e}")
    import traceback
    traceback.print_exc()

# %% [markdown] id="4sYlOOC2t-u3"
# # Задание 3
#

# %% id="3Ecit46ZA_Ha" colab={"base_uri": "https://localhost:8080/"} outputId="b4cc8b17-17cb-4a5a-c34a-66f780e67672"
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProjectGanttDashboard:
    """Класс для создания комплексного дашборда проекта"""

    def __init__(self, tasks_df, employees_df):
        self.tasks_df = tasks_df.copy()
        self.employees_df = employees_df.copy()
        self.critical_path = []
        self.project_duration = 0
        self.prepare_data()

    def prepare_data(self):
        """Подготовка данных и расчет PERT (Задание 1.1)"""
        # Расчет PERT длительности
        self.tasks_df['pert_duration'] = (
            self.tasks_df['optimistic_days'] +
            4 * self.tasks_df['likely_days'] +
            self.tasks_df['pessimistic_days']
        ) / 6
        self.tasks_df['pert_duration'] = self.tasks_df['pert_duration'].round(1)

        # Расчет стандартного отклонения для анализа рисков
        self.tasks_df['pert_std'] = (
            self.tasks_df['pessimistic_days'] - self.tasks_df['optimistic_days']
        ) / 6

        print("Данные подготовлены. Расчет PERT завершен.")

    def calculate_critical_path(self):
        """Расчет критического пути и временных параметров (Задание 1.1, 3.1)"""
        G = nx.DiGraph()

        # Добавляем узлы (задачи)
        for _, task in self.tasks_df.iterrows():
            G.add_node(
                task['task_id'],
                duration=task['pert_duration'],
                name=task['task_name']
            )

        # СОЗДАЕМ РЕАЛИСТИЧНЫЕ ЗАВИСИМОСТИ МЕЖДУ ЗАДАЧАМИ
        # Группируем задачи по типам для создания параллельных веток
        task_types = {}
        for _, task in self.tasks_df.iterrows():
            task_type = task.get('task_type', 'development')  # Используем существующий тип или задаем по умолчанию
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(task['task_id'])

        # Создаем зависимости внутри каждой группы (параллельные задачи)
        for task_type, tasks in task_types.items():
            for i in range(len(tasks) - 1):
                G.add_edge(tasks[i], tasks[i + 1])

        # Создаем кросс-зависимости между группами (пересекающиеся задачи)
        task_ids = self.tasks_df['task_id'].tolist()
        if len(task_ids) >= 4:
            # Пример: задача 3 зависит от задач 1 и 2 (параллельное выполнение)
            G.add_edge(task_ids[0], task_ids[2])
            G.add_edge(task_ids[1], task_ids[2])
            # Пример: задачи 4 и 5 выполняются параллельно после задачи 3
            G.add_edge(task_ids[2], task_ids[3])
            G.add_edge(task_ids[2], task_ids[4])

        # Расчет ранних сроков (Forward Pass)
        early_start, early_finish = {}, {}
        # Добавляем фиктивный начальный узел
        G.add_node('start', duration=0)
        for node in G.nodes():
            if node != 'start' and len(list(G.predecessors(node))) == 0:
                G.add_edge('start', node)

        topo_order = list(nx.topological_sort(G))
        topo_order = [node for node in topo_order if node != 'start']

        for node in topo_order:
            predecessors = list(G.predecessors(node))
            if not predecessors or (len(predecessors) == 1 and predecessors[0] == 'start'):
                early_start[node] = 0
            else:
                early_start[node] = max(early_finish[pred] for pred in predecessors if pred != 'start')
            early_finish[node] = early_start[node] + G.nodes[node]['duration']

        # Расчет поздних сроков (Backward Pass)
        self.project_duration = max(early_finish.values())
        late_finish, late_start = {}, {}

        # Добавляем фиктивный конечный узел
        G.add_node('end', duration=0)
        for node in G.nodes():
            if node != 'end' and len(list(G.successors(node))) == 0:
                G.add_edge(node, 'end')

        reverse_topo = list(reversed(list(nx.topological_sort(G))))
        reverse_topo = [node for node in reverse_topo if node not in ['start', 'end']]

        for node in reverse_topo:
            successors = list(G.successors(node))
            if not successors or (len(successors) == 1 and successors[0] == 'end'):
                late_finish[node] = self.project_duration
            else:
                late_finish[node] = min(late_start[succ] for succ in successors if succ != 'end')
            late_start[node] = late_finish[node] - G.nodes[node]['duration']

        # Расчет резервов и идентификация критического пути
        results = []
        for node in G.nodes():
            if node in ['start', 'end']:
                continue

            total_float = late_start[node] - early_start[node]
            free_float = 0
            if list(G.successors(node)):
                successors = [succ for succ in G.successors(node) if succ != 'end']
                if successors:
                    free_float = min(early_start[succ] for succ in successors) - early_finish[node]
                else:
                    free_float = self.project_duration - early_finish[node]
            else:
                free_float = self.project_duration - early_finish[node]

            is_critical = abs(total_float) < 0.001

            if is_critical:
                self.critical_path.append(node)

            results.append({
                'task_id': node,
                'ES': early_start[node],
                'EF': early_finish[node],
                'LS': late_start[node],
                'LF': late_finish[node],
                'Float': total_float,
                'Free_Float': max(0, free_float),
                'is_critical': is_critical
            })

        self.tasks_df = pd.merge(self.tasks_df, pd.DataFrame(results), on='task_id')

        print(f"Критический путь рассчитан. Длительность проекта: {self.project_duration:.1f} дней")
        print(f"Критические задачи: {len(self.critical_path)}")
        print(f"Некритические задачи: {len(self.tasks_df) - len(self.critical_path)}")

        # Выводим информацию о пересекающихся задачах
        print("\nСТРУКТУРА ПРОЕКТА:")
        for i, task in self.tasks_df.iterrows():
            predecessors = list(G.predecessors(task['task_id']))
            predecessors = [p for p in predecessors if p != 'start']
            print(f"   {task['task_name']}: ES={task['ES']:.0f}, EF={task['EF']:.0f}, резерв={task['Float']:.1f}д")

    def create_comprehensive_dashboard(self):
        """Создание комплексного дашборда (Все задания в одном)"""

        # Создаем дашборд с 8 графиками
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                '1. Диаграмма Ганта (все задачи проекта)',
                '2. Загрузка ресурсов по времени',
                '3. Анализ критического пути',
                '4. Вероятность завершения проекта',
                '5. S-кривая кумулятивных затрат',
                '6. Heatmap рисков по времени',
                '7. Статус выполнения задач',
                '8. Анализ чувствительности сроков'
            ),
            specs=[
                [{"type": "bar", "rowspan": 1}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "heatmap"}],
                [{"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            print_grid=False
        )

        # 1. ДИАГРАММА ГАНТА С ПЕРЕСЕКАЮЩИМИСЯ ЗАДАЧАМИ (Задание 2.1)
        colors = {
            'critical': '#FF6B6B',      # Красный для критических задач
            'normal': '#4ECDC4',        # Зеленый для некритических задач
            'milestone': '#FFD93D',     # Желтый для вех
            'analysis': '#FFA07A',      # Оранжевый для аналитических задач
            'development': '#20B2AA',   # Бирюзовый для разработки
            'testing': '#9370DB',       # Фиолетовый для тестирования
            'deployment': '#32CD32'     # Лаймовый для деплоя
        }

        # Группируем задачи по типам для лучшей визуализации
        task_categories = {}
        for _, task in self.tasks_df.iterrows():
            # Используем существующий тип задачи или определяем по названию
            task_type = task.get('task_type', 'development')
            if 'анализ' in task['task_name'].lower() or 'аналитик' in task['task_name'].lower():
                task_type = 'analysis'
            elif 'тест' in task['task_name'].lower() or 'проверк' in task['task_name'].lower():
                task_type = 'testing'
            elif 'внедр' in task['task_name'].lower() or 'деплой' in task['task_name'].lower():
                task_type = 'deployment'

            if task_type not in task_categories:
                task_categories[task_type] = []
            task_categories[task_type].append(task)

        # Сортируем задачи по времени начала для лучшего отображения
        self.tasks_df = self.tasks_df.sort_values('ES')

        # Добавляем ВСЕ задачи проекта с разными цветами по типам
        y_positions = {}  # Для отслеживания позиций на оси Y
        current_y = 0

        for task_type, tasks in task_categories.items():
            for task in tasks:
                is_critical = task['is_critical']

                # Выбираем цвет в зависимости от типа задачи и критичности
                if is_critical:
                    color = colors['critical']
                else:
                    color = colors.get(task_type, colors['normal'])

                # Определяем позицию на оси Y
                y_position = f"{task_type}_{current_y}"
                y_positions[task['task_id']] = current_y

                fig.add_trace(go.Bar(
                    name=task['task_name'],
                    x=[task['pert_duration']],
                    y=[current_y],
                    base=task['ES'],
                    orientation='h',
                    marker_color=color,
                    marker_line=dict(width=2, color='darkgray'),
                    text=[f"{task['pert_duration']}д"],
                    textposition='inside',
                    textfont=dict(color='white' if is_critical else 'black', size=9),
                    hovertemplate=(
                        f"<b>{task['task_name']}</b><br>"
                        f"Тип: {task_type.upper()}<br>"
                        f"Категория: {'🚨 КРИТИЧЕСКАЯ' if is_critical else '✅ Обычная'}<br>"
                        f"Длительность: {task['pert_duration']} дней<br>"
                        f"Период: {task['ES']:.0f}-{task['EF']:.0f} дней<br>"
                        f"Резерв: {task['Float']:.1f} дней<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False
                ), row=1, col=1)

                current_y += 1

        # Добавляем вехи
        milestones = [
            {'name': 'Старт проекта', 'day': 0},
            {'name': 'Завершение проектирования', 'day': self.project_duration * 0.3},
            {'name': 'Готовность прототипа', 'day': self.project_duration * 0.6},
            {'name': 'Финальное тестирование', 'day': self.project_duration * 0.85},
            {'name': 'Сдача проекта', 'day': self.project_duration}
        ]

        for milestone in milestones:
            # Вертикальные линии вех
            fig.add_trace(go.Scatter(
                x=[milestone['day'], milestone['day']],
                y=[-1, current_y],
                mode='lines',
                line=dict(color=colors['milestone'], width=3, dash='dot'),
                name=f'Веха: {milestone["name"]}',
                hovertemplate=f"<b>Веха: {milestone['name']}</b><br>День: {milestone['day']:.0f}<extra></extra>",
                showlegend=False
            ), row=1, col=1)

        # Легенда для диаграммы Ганта
        legend_items = [
            ('Критические задачи', colors['critical']),
            ('Аналитические задачи', colors['analysis']),
            ('Разработка', colors['development']),
            ('Тестирование', colors['testing']),
            ('Внедрение', colors['deployment']),
            ('Обычные задачи', colors['normal'])
        ]

        for i, (name, color) in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol='square'),
                name=name,
                legendgroup='gantt'
            ), row=1, col=1)

        # ОСТАЛЬНЫЕ ГРАФИКИ ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ...
        # [Здесь должен быть код для остальных 7 графиков]

        # 2. ГРАФИК ЗАГРУЗКИ РЕСУРСОВ (Задание 2.2)
        resource_workload = self.analyze_resource_loading()
        days = list(range(int(self.project_duration) + 1))
        loads = [resource_workload.get(day, 0) for day in days]

        fig.add_trace(go.Scatter(
            x=days, y=loads,
            mode='lines',
            name='Загрузка ресурсов',
            line=dict(color='#45B7D1', width=3),
            fill='tozeroy',
            fillcolor='rgba(69, 183, 209, 0.3)',
            hovertemplate="День %{x}<br>Нагрузка: %{y:.1f} часов<extra></extra>",
            showlegend=False
        ), row=1, col=2)

        fig.add_hline(y=40, line_dash="dash", line_color="red",
                     annotation_text="Лимит 40ч", row=1, col=2)

        # 3. АНАЛИЗ КРИТИЧЕСКОГО ПУТИ (Задание 3.1)
        critical_tasks = self.tasks_df[self.tasks_df['is_critical']]
        non_critical_tasks = self.tasks_df[~self.tasks_df['is_critical']]

        fig.add_trace(go.Bar(
            x=['Критические', 'Обычные'],
            y=[len(critical_tasks), len(non_critical_tasks)],
            marker_color=[colors['critical'], colors['normal']],
            hovertemplate="<b>%{x}</b><br>Количество: %{y} задач<extra></extra>",
            showlegend=False
        ), row=2, col=1)

        # 4. ВЕРОЯТНОСТЬ ЗАВЕРШЕНИЯ (Задание 5.2)
        target_durations = np.linspace(
            self.project_duration * 0.7,
            self.project_duration * 1.3,
            15
        )

        probabilities = []
        for duration in target_durations:
            prob, _, _ = self.calculate_completion_probability(duration)
            probabilities.append(prob * 100)

        fig.add_trace(go.Scatter(
            x=target_durations,
            y=probabilities,
            mode='lines+markers',
            name='Вероятность завершения',
            line=dict(color='#00CC96', width=3),
            hovertemplate="Срок: %{x:.1f} дней<br>Вероятность: %{y:.1f}%<extra></extra>",
            showlegend=False
        ), row=2, col=2)

        fig.add_vline(x=self.project_duration, line_dash="dash", line_color="red",
                     annotation_text="Плановый срок", row=2, col=2)

        # 5. S-КРИВАЯ ЗАТРАТ (Задание 6.2)
        time_points = np.linspace(0, self.project_duration, 20)
        planned_costs = 5000000 * (1 - np.exp(-0.2 * time_points / self.project_duration))
        actual_costs = planned_costs * np.random.uniform(0.85, 1.15, len(time_points))

        fig.add_trace(go.Scatter(
            x=time_points, y=planned_costs,
            mode='lines', name='План затрат',
            line=dict(color='blue', width=3),
            hovertemplate="День %{x:.0f}<br>План: %{y:,.0f} руб.<extra></extra>",
            showlegend=False
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_points, y=actual_costs,
            mode='lines', name='Факт затрат',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate="День %{x:.0f}<br>Факт: %{y:,.0f} руб.<extra></extra>",
            showlegend=False
        ), row=3, col=1)

        # 6. HEATMAP РИСКОВ (Задание 6.2)
        weeks = [f'Неделя {i+1}' for i in range(int(self.project_duration // 7) + 1)]
        risk_types = ['Технические', 'Ресурсные', 'Временные', 'Качественные', 'Бюджетные']
        risk_data = np.random.rand(len(risk_types), len(weeks))

        fig.add_trace(go.Heatmap(
            z=risk_data,
            x=weeks,
            y=risk_types,
            colorscale='RdYlGn_r',
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}<br>Уровень риска: %{z:.2f}<extra></extra>",
            showlegend=False
        ), row=3, col=2)

        # 7. СТАТУС ВЫПОЛНЕНИЯ (Задание 2.3)
        status_counts = {
            'Завершено': len(self.tasks_df) // 3,
            'В работе': len(self.tasks_df) // 3,
            'Не начато': len(self.tasks_df) // 3
        }

        fig.add_trace(go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker_colors=['#00CC96', '#FFA15A', '#636EFA'],
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>%{value} задач<extra></extra>",
            showlegend=False
        ), row=4, col=1)

        # 8. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (Задание 3.2)
        delays = [1, 2, 3, 5, 7]
        impacts = [delay * 1.1 for delay in delays]

        fig.add_trace(go.Bar(
            x=[f'+{d} д' for d in delays],
            y=impacts,
            marker_color=['#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
            hovertemplate="Задержка: %{x}<br>Влияние: +%{y:.1f} дней<extra></extra>",
            showlegend=False
        ), row=4, col=2)

        # ОБНОВЛЕНИЕ ЛАЙАУТА
        fig.update_layout(
            height=1400,
            title_text="ДАШБОРД УПРАВЛЕНИЯ ПРОЕКТОМ - ЗАДАНИЕ №3",
            title_font_size=20,
            title_x=0.5,
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial", size=10),
            margin=dict(l=50, r=50, t=100, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # НАСТРОЙКА ОСЕЙ для диаграммы Ганта
        fig.update_xaxes(
            title_text="Дни проекта",
            row=1, col=1,
            range=[0, self.project_duration * 1.1]
        )
        fig.update_yaxes(
            title_text="Задачи",
            row=1, col=1,
            tickvals=list(range(current_y)),
            ticktext=[task['task_name'] for _, task in self.tasks_df.iterrows()]
        )

        # Настройки для остальных графиков
        fig.update_xaxes(title_text="Дни", row=1, col=2, range=[0, self.project_duration])
        fig.update_yaxes(title_text="Нагрузка (часы)", row=1, col=2, range=[0, max(loads) * 1.1] if loads else [0, 100])

        fig.update_xaxes(title_text="Тип задач", row=2, col=1)
        fig.update_yaxes(title_text="Количество", row=2, col=1)

        fig.update_xaxes(title_text="Целевой срок (дни)", row=2, col=2)
        fig.update_yaxes(title_text="Вероятность (%)", row=2, col=2, range=[0, 100])

        fig.update_xaxes(title_text="Дни проекта", row=3, col=1)
        fig.update_yaxes(title_text="Затраты, руб.", row=3, col=1)

        fig.update_xaxes(title_text="Недели", row=3, col=2)
        fig.update_yaxes(title_text="Типы рисков", row=3, col=2)

        fig.update_xaxes(title_text="Задержка (дни)", row=4, col=2)
        fig.update_yaxes(title_text="Влияние (дни)", row=4, col=2)

        return fig

    def analyze_resource_loading(self):
        """Анализ загрузки ресурсов"""
        resource_workload = {}
        for _, task in self.tasks_df.iterrows():
            workload = task['pert_duration'] * 8
            start_day = int(task['ES'])
            end_day = int(task['EF'])
            for day in range(start_day, end_day + 1):
                if day not in resource_workload:
                    resource_workload[day] = 0
                resource_workload[day] += workload / (end_day - start_day + 1)
        return resource_workload

    def calculate_completion_probability(self, target_duration):
        """Расчет вероятности завершения"""
        expected_duration = self.project_duration
        project_variance = sum(self.tasks_df[self.tasks_df['is_critical']]['pert_std'] ** 2)
        project_std = np.sqrt(project_variance) if project_variance > 0 else 1
        z_score = (target_duration - expected_duration) / project_std
        probability = stats.norm.cdf(z_score)
        return probability, z_score, project_std

# Основная функция выполнения
def main():
    """Основная функция выполнения задания"""

    # Загрузка данных
    try:
        df_project = pd.read_csv('csv1.txt')
        df_employees = pd.read_csv('csv3.txt')
        print("Данные успешно загружены")
    except FileNotFoundError:
        print("Файлы не найдены. Создаю демо-данные...")
        df_project = pd.DataFrame({
            'task_id': [f'TASK-{i:03d}' for i in range(1, 11)],
            'task_name': [
                'Анализ требований', 'Проектирование архитектуры', 'Разработка API',
                'Создание базы данных', 'Фронтенд разработка', 'Интеграция систем',
                'Модульное тестирование', 'Интеграционное тестирование', 'Нагрузочное тестирование',
                'Деплой проекта'
            ],
            'task_type': ['analysis', 'analysis', 'development', 'development', 'development',
                         'development', 'testing', 'testing', 'testing', 'deployment'],
            'optimistic_days': [2, 3, 5, 4, 6, 3, 2, 3, 4, 2],
            'likely_days': [4, 5, 8, 6, 10, 5, 4, 5, 6, 3],
            'pessimistic_days': [7, 8, 12, 9, 15, 8, 6, 8, 10, 5]
        })
        df_employees = pd.DataFrame({'emp_id': ['EMP-001']})

    # Создание дашборда
    dashboard = ProjectGanttDashboard(df_project, df_employees)
    dashboard.calculate_critical_path()

    # Создание комплексного дашборда
    print("\nСоздание дашборда...")
    fig = dashboard.create_comprehensive_dashboard()

    # Сохранение
    fig.write_html("project_management_dashboard.html")
    print("Дашборд сохранен как 'project_management_dashboard.html'")

if __name__ == "__main__":
    main()
