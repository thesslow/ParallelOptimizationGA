import win32api 
import win32process 
import win32con 
import time 
import random 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from datasets import load_dataset 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process 
from GA_seri import genetic_search_cv_seri


def set_cpu_affinity():
    affinity_mask = 0b11111111 
    current_process_id = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, current_process_id)
    win32process.SetProcessAffinityMask(handle, affinity_mask)

def create_individual(param_range):
    return{
        "max_depth":random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1]) if param_ranges['max_depth'] is not None else None,
        "min_samples_split":random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1]),
        "min_samples_leaf":random.randint(param_ranges['min_samples_leaf'][0], param_ranges['min_samples_leaf'][1])
    }


def create_population_part(size, param_ranges):
    return [create_individual(param_ranges) for _ in range(size)]


def evaluate_individual(data):
    model, individual, X, y, scoring, cv = data
    model.set_params(**individual)
    scores = cross_val_score(model, X, y, scoring = scoring, cv=cv)
    
    return scores.mean()


def genetic_search_cv_prl(model, X, y, param_ranges, scoring, cv, population_size, generations, random_state = None, num_processes = cpu_count()):

    if random_state:
        random.seed(random_state)

    
    with Pool(processes=num_processes, initializer = set_cpu_affinity) as pool:
        population = []

        for generation in tqdm(range(generations), desc = "Generasi Paralel"):
            if generation == 0:

                # waktu pembuatan populasi
                start_time_population_creation = time.time()

                size_per_process = population_size // num_processes
                partial_populations = pool.starmap(
                    create_population_part,
                    [(size_per_process, param_ranges)] * num_processes
                )
                population = [ind for sublist in partial_populations for ind in sublist]

                # waktu akhir pembuatna populasi
                end_time_population_creation = time.time()
                population_creation_time_prl = end_time_population_creation - start_time_population_creation
            
            # untuk waktu paralel
            if generation == 1:
                parallel_start_time = time.time()

            data_to_evaluate = [(model, ind, X, y, scoring, cv) for ind in population]
            fitness_scores = pool.map(evaluate_individual, data_to_evaluate)

            # untuk waktu akhir akhir
            if generation == generations - 1:
                parallel_end_time = time.time()

            sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), reverse = True, key = lambda x: x[0])]
            population = sorted_population[:population_size // 2]

            offspring = []

            while len(offspring) < population_size - len(population):
                parent1, parent2 = random.sample(population, 2)
                child = {
                    "max_depth": random.choice([parent1["max_depth"], parent2["max_depth"]]),
                    "min_samples_split": random.choice([parent1["min_samples_split"], parent2["min_samples_split"]]),
                    "min_samples_leaf": random.choice([parent1["min_samples_leaf"], parent2["min_samples_leaf"]]),
                }

                if random.random() < 0.1:
                    mutate_param = random.choice(['max_depth', 'min_samples_split','min_samples_leaf'])

                    if mutate_param == "max_depth":
                        child['max_depth'] = random.randint(param_ranges['max_depth'][0],param_ranges['max_depth'][1]) if param_ranges['max_depth'] is not None else None
                    elif mutate_param == "min_samples_split":
                        child['min_samples_split'] = random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1])
                    else:
                        child["min_samples_leaf"] = random.randint(param_ranges['min_samples_leaf'][0], param_ranges['min_samples_leaf'][1])

                offspring.append(child) 
            
            population += offspring
        
        data_to_evaluate = [(model, ind, X, y , scoring, cv) for ind in population]
        fitness_scores = pool.map(evaluate_individual, data_to_evaluate)
        best_individual = population[fitness_scores.index(max(fitness_scores))]
    
    parallel_time = parallel_end_time - parallel_start_time if parallel_start_time and parallel_end_time else None
    
    return best_individual, max(fitness_scores), parallel_time, population_creation_time_prl


def run_experiment_all(num_processes, mode="parallel"):
    """
    Jalankan eksperimen serial atau paralel berdasarkan `num_processes`.
    Jika `num_processes` adalah None, eksperimen berjalan secara serial.
    """
    if mode == "serial":
        start_time_sr = time.time()
        best_params, best_scores, evaluation_time, population_creation_time_sr = genetic_search_cv_seri(
            model=clf,
            X=X_train,
            y=y_train,
            param_ranges=param_ranges,
            scoring='accuracy',
            cv=5,
            population_size=100,
            generations=100,
            random_state=42,
        )
        end_time_sr = time.time()

        sr_total_time = end_time_sr - start_time_sr

        print(f'Serial Total Time :{sr_total_time:.2f} seconds')
        return {
            "mode": "serial",
            "num_processes": 1,
            "total_time": end_time_sr - start_time_sr,
            "population_creation_time": population_creation_time_sr,
            "evaluation_time": evaluation_time,
        }
    else:
        start_time_prl = time.time()
        best_params, best_scores, parallel_time, population_creation_time_prl = genetic_search_cv_prl(
            model=clf,
            X=X_train,
            y=y_train,
            param_ranges=param_ranges,
            scoring='accuracy',
            cv=5,
            population_size=100,
            generations=100,
            random_state=42,
            num_processes=num_processes,
        )
        end_time_prl = time.time()

        prl_total_time = end_time_prl - start_time_prl

        print(f'Parallel Total Time :{prl_total_time:.2f} seconds')
        return {
            "mode": "parallel",
            "num_processes": num_processes,
            "total_time": end_time_prl - start_time_prl,
            "population_creation_time": population_creation_time_prl,
            "evaluation_time": parallel_time,
        }

# load dataset dan membuat model
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

    # paralel
    results = []
    results.append(run_experiment_all(None, mode = 'serial'))

    for n in [2,4,6,8]:
        results.append(run_experiment_all(n, mode = 'parallel'))
    

    df_results = pd.DataFrame(results)

    df_results.to_csv("GA_experimetns_all.csv", index = True)

    # Visualisasikan hasil
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["num_processes"], df_results["total_time"], marker='o', label="Total Time")
    plt.plot(df_results["num_processes"], df_results["population_creation_time"], marker='o', label="Population Creation Time")
    plt.plot(df_results["num_processes"], df_results["evaluation_time"], marker='o', label="Fitness Evaluation Time")

    plt.xlabel("Number of Processes")
    plt.ylabel("Time (seconds)")

    x_tick_labels = ["Serial" if x == 1 else str(f'{x} Proses') for x in df_results['num_processes']]
    plt.xticks(df_results['num_processes'], x_tick_labels)

    plt.title("Execution Time: Serial vs Parallel")
    plt.legend()
    plt.grid(True)
    plt.show()