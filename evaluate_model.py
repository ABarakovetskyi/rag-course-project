from evaluate import load
from app import get_response

# Завантаження метрик
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Підготовка еталонних відповідей
test_pairs = [
    #("Які завдання за напрямом основної роботи у начальника сектора СЕД та УП?",
     #"Проектує, впроваджує, реалізує та підтримує функціонування автоматизованих систем управління виробничими процесами у ПрАТ «Львівобленерго» на базі систем електронного документообігу iDoc, систем управління персоналом SAP R/3, електронної бібліотеки та інших."),

    #("Які обов’язки щодо охорони праці має начальник сектора СЕД та УП?",
     #"Начальник сектора СЕД та УП співпрацює у справі організації безпечних і нешкідливих умов праці, вживає заходів щодо усунення виробничих ситуацій, що створюють загрозу життю чи здоров’ю працівників."),

    #("Яка кваліфікація (ступінь освіти, стаж, категорія) потрібна для психолога 1 категорії згідно посадової інструкції?",
     #"Посаду Психолога 1 категорії може обіймати особа з повною вищою освітою відповідного напряму підготовки (спеціаліст чи магістр). Стаж роботи за професією психолога 2 категорії – не менше 2 років."),

    ("Які основні завдання психолога 1 категорії?",
     "Психолог 1 категорії проводить психофізіологічну експертизу працівників, які виконують роботи підвищеної небезпеки, проводить тренінги корекції функціонального стану працівників, визначає психофізіологічні якості, що впливають на працездатність."),

    #("Яке навчання з охорони праці повинні проходити працівники?",
     #"Положення встановлює види і порядок проведення навчання, перевірки знань, інструктажів, стажування, дублювання, тренажерної підготовки та допуску до роботи з питань охорони праці, цивільного захисту та пожежної безпеки."),

    #("Які завдання передбачені для спеціального навчання з пожежної безпеки?",
     #"Навчання з пожежної безпеки охоплює знання нормативно-правових актів, правил поводження з устаткуванням, користування засобами пожежогасіння та проходження медичних оглядів.")
]

# Підготовка даних для обчислення метрик
references = []
predictions = []

for (query, ref) in test_pairs:
    hyp, _ = get_response(query)  # отримуємо відповідь від моделі

    predictions.append(hyp)  # просто рядок без вкладення в список
    references.append([ref])  # лише один рівень вкладеності

# Додати перевірку отриманих відповідей
for i, (query, ref) in enumerate(test_pairs):
    print(f"Запит {i + 1}: {query}")
    print(f"Очікувана відповідь: {ref}")
    print(f"Отримана відповідь: {predictions[i]}")
    print("---")


# Обчислення BLEU
bleu_result = bleu_metric.compute(predictions=predictions, references=references)
print("BLEU:", bleu_result)

# Обчислення ROUGE
rouge_result = rouge_metric.compute(predictions=predictions, references=references)
print("ROUGE:", rouge_result)
