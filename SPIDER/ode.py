import os
import copy
import torch
import random
import pickle
import numpy as np
from model import Model
from ops import OPS_Keys
from utils.distances import *
import torch.utils.data as data
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from utils.spider_loader import SPIDERDataset
import torchvision.transforms as transforms

class DE():
    
    def __init__(self, pop_size = None, 
                 mutation_factor = None, 
                 crossover_prob = None, 
                 boundary_fix_type = 'random', 
                 seed = None,
                 mutation_strategy = 'rand1',
                 crossover_strategy = 'bin'):

        # DE related variables
        self.NP = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.boundary_fix_type = boundary_fix_type

        # Global trackers
        self.P_G = []
        self.P0 = [] # P0 population
        self.OP0 = [] # Opposite of the P0 population
        self.history = []
        self.allModels = dict()
        self.best_arch = None
        self.seed = seed

        # CONSTANTS
        self.FE_max = 250
        self.NUM_EDGES = 9          
        self.NUM_VERTICES = 6       #güncel düğüm sayısı  input + 4 düğüm + output
        self.DIMENSIONS = 16        #vektör boyutu 16 olarak güncelendi
        self.MAX_STACK = 3
        self.MAX_NUM_CELL = 3
        self.JUMPING_RATE = 0.3
        self.STACKS = [i for i in range(1, self.MAX_STACK + 1)] # 1, 2, 3
        self.CELLS = [i for i in range(1, self.MAX_NUM_CELL + 1)] # 1, 2, 3
        self.NBR_FILTERS = [2**i for i in range(5, 8)] # 32, 64, 128
        #self.NBR_FILTERS = [2**i for i in range(3, 6)] # 8, 16, 32
        self.OPS = copy.deepcopy(OPS_Keys)
    
    # kaotik başlangıç dizisi phi_z' i güncellemek ve kaotik popülasyon üretmek için circle map kullanılmıştır.
    def circle_map(self, phi_z):         
        return np.mod(phi_z + 0.2 - (0.5 / (2 * np.pi)) * np.sin(2 * np.pi * phi_z), 1.0 )
    
    def reset(self):
        self.best_arch = None
        self.P_G = []
        self.P0 = []
        self.OP0 = []
        self.allModels = dict()
        self.history = []
        self.init_rnd_nbr_generators()
    
    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.jumping_rnd = np.random.RandomState(self.seed)

    def seed_torch(self, seed=None):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/{data_flag}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)


    def init_chaotic_P0_population(self, pop_size=None):
        i = 0
        Z = 10
        D = self.DIMENSIONS
        
        for i in range(pop_size):
            X_i = np.zeros(D)
            for d in range(D):
                z = 0
                phi_z = random.random()
                while z <= Z:
                    phi_z = self.circle_map(phi_z) 
                    z += 1
                
                X_i[d] = phi_z * (1 - 0) + 0
            
            # Chromosome and Model Creation
            chromosome = X_i.copy()
            config = self.vector_to_config(chromosome)
            X_i_P = Model(chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES)

            # Same Solution Check
            isSame, _ = self.checkSolution(X_i_P)
            if not isSame:
                X_i_P.solNo = self.solNo
                self.solNo += 1
                self.allModels[X_i_P.solNo] = {"org_matrix": X_i_P.org_matrix.astype("int8"), 
                                               "org_ops": X_i_P.org_ops,
                                               "chromosome": X_i_P.chromosome,
                                               "fitness": X_i_P.fitness}                                               
                self.P0.append(X_i_P)
                self.writePickle(X_i_P, X_i_P.solNo)
                i += 1


    def get_opposite_model(self, model, a = 0, b = 1):

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            opposite_chromosome = np.array([a[idx] + b[idx] - c for idx, c in enumerate(model.chromosome)])
        else:
            opposite_chromosome = np.array([a + b - c for c in model.chromosome])
        
        config = self.vector_to_config(opposite_chromosome)
        opposite_model = Model(opposite_chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES)
        
        return opposite_model


    def init_chaotic_OP0_population(self):
        counter = 0

        while counter < len(self.P0):
            X_i_OP = self.get_opposite_model(self.P0[counter])
            # Same Solution Check
            isSame, _ = self.checkSolution(X_i_OP)
            if not isSame:
                self.solNo += 1
                X_i_OP.solNo = self.solNo
                self.allModels[X_i_OP.solNo] = {"org_matrix": X_i_OP.org_matrix.astype("int8"), 
                                                        "org_ops": X_i_OP.org_ops,
                                                        "chromosome": X_i_OP.chromosome,
                                                        "fitness": X_i_OP.fitness}
                self.OP0.append(X_i_OP)
                self.writePickle(X_i_OP, X_i_OP.solNo)
            counter += 1

            


    def checkSolution(self, model):
        model_dict = {"org_matrix": model.org_matrix.astype("int8"), 
                      "org_ops": model.org_ops}
        for i in self.allModels.keys():
            model_2 = self.allModels[i]
            D = jackard_distance_caz(model_dict, model_2)
            if D == 0:
                return True, model_2
        
        return False, None          
    
    def sample_population(self, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.P_G)), size, replace=False)
        return self.P_G[selection]
    
    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        
        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector

    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            config = np.zeros(self.DIMENSIONS, dtype='uint8')
            
            max_edges = int((self.NUM_VERTICES * (self.NUM_VERTICES - 3)) / 2)    #9

            #EDGES
            # Hücreler arası bağlantıları belirleme
            node_connections = {
                1: [2, 3, 4, 5, 6],
                2: [3, 4, 5, 6, 0],
                3: [4, 5, 6, 0],
                4: [5, 6, 0],
                5: [6, 0]
            }

            for idx in range(max_edges):
                # Düğüm ve bağlantılar
                node = list(node_connections.keys())[idx // 2]       #mevcut düğüm
                connections = node_connections[node]                 #mevcut düğümün bağlantı olasılıkları
                num_options = len(connections)                       
                
                config[idx] = connections[self.get_param_value(vector[idx], num_options)]
                
            # Vertices - Ops 
            for idx in range(max_edges, max_edges + 4):  #9, 10, 11, 12
                config[idx] = self.get_param_value(vector[idx], len(self.OPS))

            # Number of Cells
            idx = max_edges + self.NUM_VERTICES - 2  #13
            config[idx] = self.get_param_value(vector[idx], len(self.CELLS))

            
            # Number of Stacks
            config[idx + 1] = self.get_param_value(vector[idx + 1], len(self.STACKS)) #14

            # Number of Filters
            config[idx + 2] = self.get_param_value(vector[idx + 2], len(self.NBR_FILTERS))  #15

        except:
            print("HATA...", vector)

        return config

    def f_objective(self, model):
        if model.isFeasible == False: # Feasibility Check
            return -1, -1
        
        # Else  
        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        if fitness != -1:
            self.FE += 1
            self.allModels.setdefault(model.solNo, dict())
            self.allModels[model.solNo]["fitness"] = fitness
            with open(f"results/{data_flag}/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness, cost


    def init_eval_pop(self):
        '''
          Bu bölümde kaotik popülasyon ve karşıt kaotik popülasyon bireylerinin her biri için fitness değeri hesaplanır ve popülasyonlar birleştirilir, 
          elde edilen birleşim kümesindeki bireylerden fitness değeri en iyi olanlar sıralanır,
          pop_size büyüklüğünde yeni bir popülasyon oluşturulur.
        '''
        print("Start Initialization...")

        # Kaotik ve Karşıt KAotik popülasyonlar oluşturulur
        self.init_chaotic_P0_population(self.NP)
        self.init_chaotic_OP0_population()

        # Kaotik popülasyon bireylerinin her biri için fitness değeri hesaplanır
        for X_i_P in self.P0:
            X_i_P.fitness, cost = self.f_objective(X_i_P)
            self.writePickle(X_i_P, X_i_P.solNo)

        # Karşıt Kaotik popülasyon bireylerinin her biri için fitness değeri hesaplanır
        for X_i_OP in self.OP0:
            X_i_OP.fitness, cost = self.f_objective(X_i_OP)
            self.writePickle(X_i_OP, X_i_OP.solNo)
        
        # Popülasyonlar birleştirilir
        self.P0.extend(self.OP0)
        self.P_G = sorted(self.P0, key = lambda x: x.fitness, reverse=True)[:self.NP]
        self.best_arch = self.P_G[0]

        del self.P0
        del self.OP0
        
        return np.array(self.P_G)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant


    # adaptive mutasyon stratejileri
    """
    DE/current-to-pbest/1 mutasyon stratejisi:
    V_i = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)

    Parametreler:
    - i: Mutasyona uğrayacak bireyin indeksi
    - p: En iyi %p bireylerden biri seçilir (0.2 değeri superior bireyler düşünülerek seçildi.)
    """
    def mutation_current_to_pbest(self, i, p=0.2):
        X_i = copy.deepcopy(self.P_G[i].chromosome)                 # mevcut çözüm vektörü
        
        p_best_count = max(1, int(self.NP * p))      # en iyi uygunluk değerine sahip 
        p_best_subset = self.P_G[:p_best_count]     # ilk %p lik kümedeki bireylerden biri 
        Xp_best = random.choice(p_best_subset).chromosome  # rastgele seçilir. (Xp_best)

        indices = list(range(self.NP))               # mevcut çözüm haricinde rastgele seçilen   
        indices.remove(i)                                  # farklı iki birey
        r1, r2 = random.sample(indices, 2)

        X_r1 = self.P_G[r1].chromosome               # seçilen bireylerin kromozomları alınır.
        X_r2 = self.P_G[r2].chromosome

        F = self.mutation_factor
        mutant = X_i + F * (Xp_best - X_i) + F * (X_r1 - X_r2)

        return mutant
    """
    DE/current-to-rand/1 mutasyon stratejisi:
    V_i = X_i + rand * (X_r1 - X_i) + F * (X_r2 - X_r3)
    
    Parametre:
    - i: Mutasyona uğrayacak bireyin indeksi
    """
    def mutation_current_to_rand(self, i):
        X_i = copy.deepcopy(self.P_G[i].chromosome)          # mevcut çözüm vektörü

        indices = list(range(self.NP))        # mevcut çözüm haricinde rastgele seçilen
        indices.remove(i)                           
        r1, r2, r3 = random.sample(indices, 3)      # üç farklı birey

        X_r1 = self.P_G[r1].chromosome
        X_r2 = self.P_G[r2].chromosome
        X_r3 = self.P_G[r3].chromosome

        rand_number = np.random.rand()
        F = self.mutation_factor

        mutant = X_i + rand_number * (X_r1 - X_i) + F * (X_r2 - X_r3)

        return mutant
    
    # adaptive mutasyon için fonksiyon kodu

    def adaptive_mutation(self, i, mode):
        # Superior ve Inferior bireylerin sayıları belirlendi
        superior_count = int(self.NP * 0.2)
        inferior_count = int(self.NP * (0.5 - 0.005 * 10 ** (2 * self.FE / self.FE_max)))

        # Bireyin durumu belirlendi ve duruma göre uygun mutasyon stratejisi gerçekleştirildi
        if i <= superior_count:
            # Superior bireyler için: DE/current-to-pbest/1
            return self.mutation_current_to_pbest(i)
    
        elif i <= (superior_count + self.NP - (superior_count + inferior_count)):
            # Normal bireyler için: mode'a göre seçim
            if mode == "exploration":
                return self.mutation_current_to_rand(i)
            else:  # exploitation
                return self.mutation_current_to_pbest(i)
    
        else:
            # Inferior bireyler için: DE/current-to-rand/1
            return self.mutation_current_to_rand(i)

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1.chromosome, r2.chromosome, r3.chromosome)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome, r5.chromosome)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best, r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.chromosome, best.chromosome, r2.chromosome, r3.chromosome)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n+L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring
    
    def readPickleFile(self, file):
        with open(f"results/model_{file}.pkl", "rb") as f:
            data = pickle.load(f)
        
        return data

    # adaptive mutasyon stratejisi için güncellemelere buradan başlanıldı. 
    # sözde kodun 21. satırından itibaren

    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        P_T = []    # Next population

        # 22. Popülasyonu fitness’e göre azacak şekilde sıralanması
        self.P_G = sorted(self.P_G, key=lambda x: x.fitness, reverse=True)

        # 23-24. Superior ve Inferior bireylerin belirlenmesi
        superior_count = int(self.NP * 0.2)
        inferior_count = int(self.NP * (0.5 - 0.005 * 10 ** (2 * self.FE / self.FE_max)))

        # 25-31. Ortalama uzaklık hesaplaması (sadece normal bireyler üzerinden)
        D_avg = 0
        distances = []      #mesafe listesi tanımlandı

        for i in range(self.NP):
            if i > superior_count and i <= (superior_count + self.NP - (superior_count + inferior_count)):
                X_i = self.P_G[i].chromosome
                X_best = self.P_G[0].chromosome
                D_best = np.linalg.norm(X_i - X_best)         # xi ve xbest arasındaki euclidean uzaklığı hesaplanır
                distances.append(D_best)                    # tüm normal bireylerin en iyi bireye olan mesafeleri distance kümesine eklenir
                D_avg = D_avg + D_best

        # 32.
        D_avg_plus = 0                                      # mesafesi, ortalama uzaklıktan fazla olan bireylerin sayısı
        
        # 33. Davg hesaplanması
        if len(distances) > 0:
            D_avg = D_avg / len(distances)    #?? sözde kodda değiştirilecek
        else:
            D_avg = 0

        # 34-36. Davg'den büyük olan bireylerin sayısını bul
        for d in distances:
            if d > D_avg:
                D_avg_plus += 1

        # 37- 40. Mode: exploration mı, exploitation mı?
        if D_avg_plus > len(distances) * 0.5:   #?? sözde kodda değiştirilecek
            mode = "exploration"
        else:
            mode = "exploitation"

        # 42-49. # mutation -> crossover
        for j in range(self.NP):
            Xi = copy.deepcopy(self.P_G[j].chromosome)

            Vi = self.adaptive_mutation(j, mode)
            Ui = self.crossover(Xi, Vi)
            Ui = self.boundary_check(Ui)

            config = self.vector_to_config(Ui)
            model = Model(Ui, config,
                          self.CELLS[config[-3]],
                          self.STACKS[config[-2]],
                          self.NBR_FILTERS[config[-1]],
                          NUM_CLASSES)
            self.solNo += 1
            model.solNo = self.solNo
            trials.append(model)

        trials = np.array(trials)

        # selection
        for j in range(self.NP):
            target = self.P_G[j]
            mutant = trials[j]

            isSameSolution, sol = self.checkSolution(mutant)
            if isSameSolution:
                print("SAME SOLUTION")
                cfg = self.vector_to_config(sol["chromosome"])
                mutant = Model(sol["chromosome"], cfg,
                               self.CELLS[cfg[-3]],
                               self.STACKS[cfg[-2]],
                               self.NBR_FILTERS[cfg[-1]],
                               NUM_CLASSES)
                mutant.fitness = sol["fitness"]
            else:
                self.f_objective(mutant)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = {
                    "org_matrix": mutant.org_matrix.astype("int8"),
                    "org_ops": mutant.org_ops,
                    "chromosome": mutant.chromosome,
                    "fitness": mutant.fitness
                }

            # Check Termination Condition
            if self.FE > self.FE_max:
                return

            if mutant.fitness >= target.fitness:
                P_T.append(mutant)
                del target
                if mutant.fitness >= self.best_arch.fitness:
                    self.best_arch = mutant
            else:
                P_T.append(target)
                del mutant

        self.P_G = P_T

        ## Opposition-Based Generation Jumping
        if self.jumping_rnd.uniform() < self.JUMPING_RATE:
            chromosomes = []
            for X_i_P in self.P_G:
                chromosomes.append(X_i_P.chromosome)
            
            min_p_j = np.min(chromosomes, 0)
            max_p_j = np.max(chromosomes, 0)

            counter = 0
            while counter < self.NP:
                X_i_OP = self.get_opposite_model(self.P_G[counter], a = min_p_j, b = max_p_j)
                # Same Solution Check
                isSame, _ = self.checkSolution(X_i_OP)
                if not isSame:
                    self.solNo += 1
                    X_i_OP.solNo = self.solNo
                    self.f_objective(X_i_OP)
                    self.allModels[X_i_OP.solNo] = {"org_matrix": X_i_OP.org_matrix.astype("int8"), 
                                                            "org_ops": X_i_OP.org_ops,
                                                            "chromosome": X_i_OP.chromosome,
                                                            "fitness": X_i_OP.fitness}
                    self.P_G.append(X_i_OP)
                    self.writePickle(X_i_OP, X_i_OP.solNo)
                counter += 1
            
            self.P_G = sorted(self.P_G, key = lambda x: x.fitness, reverse=True)[:self.NP]
            if self.P_G[0].fitness >= self.best_arch.fitness:
                self.best_arch = self.P_G[0]

        self.P_G = np.array(self.P_G)

    def run(self, seed):
        self.seed = seed
        self.solNo = 0
        
        self.G = 0
        self.FE = 0

        self.reset()
        self.seed_torch(seed=self.seed)
        self.P_G = self.init_eval_pop()

        while self.FE < self.FE_max:
            self.evolve_generation()
            print(f"Generation:{self.G}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
            self.G += 1     
        
if __name__ == "__main__":
    device = torch.device('cuda:1')

    result_path = "SPIDER"
    data_flag = f"SPIDER"
    if os.path.exists(f"results/{data_flag}/") == False:
        os.makedirs(f"results/{data_flag}/")

    seed = 42
    random.seed(seed)
    
    NUM_CLASSES = 13
    BATCH_SIZE = 64

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset instances for all splits
    train_dataset = SPIDERDataset(
        data_dir="SPIDER-colorectal/SPIDER-colorectal", 
        context_size=1, 
        split="train", 
        transform=data_transform,
        nas_stage=True,
        percentage=0.1
    )
    
    val_dataset = SPIDERDataset(
        data_dir="SPIDER-colorectal/SPIDER-colorectal", 
        context_size=1, 
        split="val", 
        transform=data_transform,
        records=train_dataset.val_records,
        nas_stage=True,
        percentage=0.1
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1)
    metric_fn.cuda(device)

    de = DE(pop_size=20, mutation_factor=0.5, crossover_prob=0.5, seed=seed)
    de.run(seed)