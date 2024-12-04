import time 
import random 
import pandas as pd 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from datasets import load_dataset 
from tqdm import tqdm

def create_individual(param_ranges):
    return {
        "max_depth":random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1]) if param_ranges['max_depth'] is not None else None,
        "min_samples_split":random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1]),
        "min_samples_leaf":random.randint(param_ranges['min_samples_leaf'][0], param_ranges['min_samples_leaf'][1])
    }



def create_partial_population(size, param_ranges):
    return [create_individual(param_ranges) for _ in range(size)]

def genetic_search_cv_seri(
    model, X, y, param_ranges, scoring, cv, population_size, generations, random_state = None
):
    if random_state:
        random.seed(random_state)

    def evaluate_individual(individual):
        model.set_params(**individual)
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        return scores.mean()

    # waktu pembuatan populasi seri
    start_time_population_creation_sr = time.time()
    population = [create_individual(param_ranges) for _ in range(population_size)]

    # waktu akhir pembuatan populasi seri
    end_time_population_creation_sr = time.time()
    population_creation_time_sr = end_time_population_creation_sr - start_time_population_creation_sr

    evaluation_time = 0

    for generation in tqdm(range(generations), desc = "Generasi Seri"):

        # waktu start untuk seri
        start_evaluation_time = time.perf_counter()

        fitness_scores = [evaluate_individual(ind) for ind in population]

        # akhir waktu evaluasi
        end_evaluation_time = time.perf_counter()
        evaluation_time += end_evaluation_time - start_evaluation_time

        sorted_population = [ind for _, ind in sorted(zip(fitness_scores,population), reverse = True, key = lambda x: x[0])]


        population = sorted_population[:population_size // 2]

        offspring = []
        while len(offspring) < population_size - len(population):
            parent1, parent2 = random.sample(population, 2)
            child = {
                "max_depth":random.choice([parent1['max_depth'], parent2['max_depth']]),
                "min_samples_split":random.choice([parent1['min_samples_split'], parent2['min_samples_split']]),
                "min_samples_leaf":random.choice([parent1['min_samples_leaf'], parent2['min_samples_leaf']])
            }

            if random.random() < 0.1:
                mutate_param = random.choice(['max_depth','min_samples_split','min_samples_leaf'])
                if mutate_param == "max_depth":
                    child['max_depth'] = random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1])
                elif mutate_param == "min_samples_split":
                    child['min_samples_split'] = random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1])
                else:
                    child['min_samples_leaf'] = random.randint(param_ranges['min_samples_leaf'][0], param_ranges['min_samples_leaf'][1])

            
            offspring.append(child)
        
        population += offspring

    fitness_scores = [evaluate_individual(ind) for ind in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]

    return best_individual, max(fitness_scores), evaluation_time, population_creation_time_sr

    
    

dataset = load_dataset("julien-c/titanic-survival")

df = pd.DataFrame(dataset['train'])
df = df.drop(columns = ['Name'])

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Siblings/Spouses Aboard'] = df['Siblings/Spouses Aboard'].fillna(df['Siblings/Spouses Aboard'].mode()[0])
df['Parents/Children Aboard'] = df['Parents/Children Aboard'].fillna(df['Parents/Children Aboard'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].mode()[0])
df['Pclass'] = df['Pclass'].fillna(df['Pclass'].mode()[0])


df = pd.get_dummies(df, columns=['Sex'], drop_first = True)

X = df.drop(columns = ['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2 , random_state = 42)

param_ranges = {
    "max_depth":[2,100],
    "min_samples_split":[2,100],
    "min_samples_leaf":[2,100]
}

clf = DecisionTreeClassifier(random_state = 42)

if __name__ == "__main__":
    print("=======================seri======================================================================")

    start_time = time.time()

    best_params, best_scores, evaluation_time, population_creation_time_sr = genetic_search_cv_seri(
        model = clf,
        X = X_train,
        y = y_train,
        param_ranges = param_ranges,
        scoring = 'accuracy',
        cv = 5,
        population_size = 100, 
        generations = 100,
        random_state = 42
    )

    

    print(f'Best Parameter {best_params}')
    print(f'Best Score form cross_validation : {best_scores:.2f}')

    clf.set_params(**best_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Akurasi pada data uji: {accuracy:.2f}')


    print(f'Serial Population Creation Time : {population_creation_time_sr:.100f}')
    print(f"Serial Fitness Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Total Execution Time (Incl. Serial Overheads): {elapsed_time:.2f} seconds")
    print("=======================seri======================================================================")

    
    # print(f"Paralel Fitness Evaluation")
    # print(f"Start Time: {start_time}")
    # print(f"End Time: {end_time}")
    # print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    # print("=======================seri======================================================================")

