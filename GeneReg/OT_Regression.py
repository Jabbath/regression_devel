import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import anndata
from io import BytesIO
from sklearn import linear_model
import pickle
from datetime import datetime
from kneed import KneeLocator

class Fate_Lasso():
    
    #Initializes Fate_Lasso with the following:
    #adata_full: An anndata containing all the cells given in fates
    #fates: A pandas df containing cells on the rows and cell types
    #on the columns. Entries are fate probabilities for each cell type.
    #type_names: The type_names that should be examined for the regression
    #TODO: make type_names optional and make it all the non-zero columns of fates
    #when not specified
    #factor_genes: Transcription factor genes which we'll restrict the regression to
    #common_names: The common names of genes
    #indices: The indices of the cells we will consider
    def __init__(self, adata_full, fates, type_names,
                 factor_genes, common_names, indices):
        self.adata_full = adata_full
        
        #Copy the fates so we don't modify the original
        fates_c = fates.copy()
        self.fates = fates_c
        
        self.type_names = type_names
        self.factor_genes = factor_genes
        self.common_names = common_names
        
        self.TRAIN_RATIO = 0.7
        self.train, self.test = self.split_data(indices, self.TRAIN_RATIO)
        
        #Memoize previously computed fits
        self.computed_fits = {cell_type: {} for cell_type in type_names}
        
   #################################################
   # Core regression code
   #################################################
    # Given a list of cells which comprise our data,
    # and a training ratio, randomly splits the cells
    # into training and testing sets. training_ratio of
    # the cells should fall into the training set.
    def split_data(self, cells, training_ratio):    
        # Shuffle the indices
        indices = list(cells)
        np.random.shuffle(indices)

        # Split the data
        end_index = int(len(indices) * training_ratio)
        train = indices[:end_index]
        test = indices[end_index:]

        return train, test


    def train_type_at_hpf(self, cell_type, alpha, positive=False):    
        #Get training expression data and bary coords
        train_exp_data = self.adata_full[self.train, self.factor_genes].X
        train_bary_data = self.fates.loc[self.train, cell_type]
        train_bary_data = train_bary_data.to_numpy()
        
        #Train the model, checking if we already computed for this alpha
        if str(alpha) in self.computed_fits[cell_type]:
            reg = self.computed_fits[cell_type][str(alpha)]
        else:
            reg = linear_model.Lasso(alpha=alpha, positive=positive)
            reg.fit(train_exp_data, train_bary_data)
            
            self.computed_fits[cell_type][str(alpha)] = reg
      
        coeffs = reg.coef_
        intercept = reg.intercept_
        
        #Get R^2 score
        R2 = reg.score(self.adata_full[self.test, self.factor_genes].X, self.fates.loc[self.test, cell_type])

        return coeffs, R2, intercept
    
    #Given a list of cells, trains against all possible cell types
    #Returns the found coefficients arranged by cell type, the R^2
    #value by cell type, and the training and testing sets of cells used.
    def train_cells(self, alphas, positive=False):
        #Store the results of each fit
        coeffs_arr = {cell_type: [] for cell_type in self.type_names}
        R2_arr = {cell_type: -9999 for cell_type in self.type_names}
        intercepts = {cell_type: -9999 for cell_type in self.type_names}
        
        for cell_type, alpha in zip(self.type_names, alphas):
            coeffs, R2, intercept = self.train_type_at_hpf(cell_type, alpha, positive=positive)
            coeffs_arr[cell_type] = coeffs
            R2_arr[cell_type] = R2
            intercepts[cell_type] = intercept
            print('Finished training type =  ', cell_type)

        return coeffs_arr, R2_arr, intercepts
    
    #Given a list of fit coefficients for each cell type
    #and an array of R^2 values for each cell type,
    #creates a list of genes and corresponding coefficients
    def make_gene_lists(self, coeffs_arr, R2_arr, intercepts):
        #Make a list of genes in the network for each cell type (formatted without the P())
        genes_lists = {cell_type: {'genes': [], 'coefficients': [], 
                                   'R^2': -9999, 'intercept': -9999} for cell_type in self.type_names}

        for cell_type in self.type_names:
            print('Found genes for ', cell_type)
            coeffs = coeffs_arr[cell_type]
            genes = self.factor_genes

            print('Number of non-zero genes: ', np.count_nonzero(coeffs))
            for i in range(len(coeffs)):
                coeff = coeffs[i]

                if(coeff != 0):
                    genes_lists[cell_type]['genes'].append(genes[i])
                    genes_lists[cell_type]['coefficients'].append(coeff)
        
        #Add R^2 values and intercepts
        for cell_type in self.type_names:
            R2 = R2_arr[cell_type]
            genes_lists[cell_type]['R^2'] = R2
            
            intercept = intercepts[cell_type]
            genes_lists[cell_type]['intercept'] = intercept

        return genes_lists
    
    #Given the indices of test cells and an array of R^2 values
    #prints metrics about the quality of fit. These currently include
    #avg. residual, std. dev. of residuals, R^2
    def analyze_fit(self, R2_arr):
        #Go through each cell type and calculate the residual
        for cell_type in self.type_names:
            score = R2_arr[cell_type]
            print('R^2 for {} is:'.format(cell_type), score)
            
    def train_analyze(self, alphas, positive=False):
        #Run training
        coeffs_arr, R2_arr, intercepts = self.train_cells(alphas, positive=positive)
        print()
        #Make gene lists
        genes_list = self.make_gene_lists(coeffs_arr, R2_arr, intercepts)
        print()
        #Print stats about the fit
        self.analyze_fit(R2_arr)
        return genes_list
    
    ##########################################################
    # Regression code with support for first encounter time
    ##########################################################
    
    #Given cells to train/test on and alphas to check,
    #computes gene lists for the cells at each alpha
    def get_gene_lists_at_alphas(self, alphas, positive=False):
        gene_lists = []

        #Go through each alpha and train each type at that alpha
        for alpha in alphas:
            alphas_arr = [alpha for i in range(10)]
            genes_list = self.train_analyze(alphas_arr, positive=positive)

            gene_lists.append(genes_list)

        return gene_lists
    
    #Given gene lists computed at different alphas, a cell type,
    #and a gene, computes the number of the first list where that
    #gene is encountered. Throws error if the gene is not found.
    def get_first_encounter_num(self, gene_lists, cell_type, gene):
        first_encounter = -1

        #Go through each gene list and look for the gene
        for i, gene_list in enumerate(gene_lists):
            if gene in gene_list[cell_type]['genes']:
                first_encounter = i
                break

        #If the gene wasn't found throw an err
        assert first_encounter >= 0

        return first_encounter

    #Given gene lists computed at different alphas, a cell type,
    #and a list of genes, computes the number of the first gene list
    #that gene is encountered for each gene.
    def get_first_encounters_geneset(self, gene_lists, cell_type, genes):
        first_encounter_nums = []

        #Get a first encounter number for each gene
        for gene in genes:
            first_encounter_num = self.get_first_encounter_num(gene_lists, cell_type, gene)
            first_encounter_nums.append(first_encounter_num)

        return first_encounter_nums

    #Given gene lists computed at each alpha in alphas, and a target alpha and cell
    #type, returns the gene list for that target alpha and cell type.
    def get_gene_list_from_gene_lists(self, gene_lists, cell_type, target_alpha, alphas):
        #When we could not get a fit, just return an empty list
        if target_alpha < 0:
            return {'genes': [], 'coefficients': []}
            
        target_index = alphas.index(target_alpha)
        gene_list = gene_lists[target_index][cell_type]
        return gene_list


    #Given a set of gene_lists computed for each alpha in alphas
    #and a set of target alphas for each cell type. The target alphas
    #represent which alpha we want to use to base our first encounter
    #rank off of for each cell type. 
    def get_first_encounter_ranks(self, gene_lists, target_alphas, alphas):
        final_gene_list = {}

        for cell_type, target_alpha in zip(self.type_names, target_alphas):
            #Get the gene list for our cell type, computed for the target alpha
            gene_list = self.get_gene_list_from_gene_lists(gene_lists, cell_type, target_alpha, alphas)

            #Get the first encounter rank for each gene in the gene list
            first_encounters = self.get_first_encounters_geneset(gene_lists, cell_type, gene_list['genes'])
            gene_list['First Encounter'] = first_encounters

            final_gene_list[cell_type] = gene_list

        return final_gene_list
    
    #Given a set of gene_lists computed at mulitple alphas,
    #and a pandas df of ATG names to common names,
    #converts the ATG names to common
    def lists_to_common_names(self, gene_lists, common_names):
        #Go through each alpha and change genes for each cell type
        for gene_list in gene_lists:
            for cell_type in self.type_names:
                ATG_names = gene_list[cell_type]['genes']
                common = list(self.common_names.loc[ATG_names].Name)
                gene_list[cell_type]['genes'] = common

        return gene_lists
    
    ##########################################################
    # Plotting code that can be used to determine default
    # values for alpha
    ##########################################################
    
    #Given a list of numbers of genes, scores, and alphas, filters
    #the number of genes list for repeat values and removes
    #those indices from the other two arrays.
    def filter_for_repeats(self, num_genes_arr, scores, alphas):
        num_genes_arr_filt = num_genes_arr.copy()
        scores_filt = scores.copy()
        alphas_filt = alphas.copy()

        #Track repeated values
        num_genes_to_remove = []
        scores_to_remove = []
        alphas_to_remove = []

        for i in range(len(num_genes_arr_filt) - 1):
            if num_genes_arr_filt[i] == num_genes_arr_filt[i+1]:                
                #Store the values we need to remove if there's a repeat
                num_genes_to_remove.append(num_genes_arr_filt[i])
                scores_to_remove.append(scores_filt[i])
                alphas_to_remove.append(alphas_filt[i])

        for num_genes, score, alpha in zip(num_genes_to_remove, 
                                            scores_to_remove, alphas_to_remove):
            num_genes_arr_filt.remove(num_genes)
            scores_filt.remove(score)
            alphas_filt.remove(alpha)

        return num_genes_arr_filt, scores_filt, alphas_filt
    
    #Given cells to fit on and alphas, creates plots
    #of R^2 vs. Number of genes calculated at each alpha,
    #for each cell type. Returns the alphas at the knee point
    #for each cell type.
    def make_plots_R2_num_genes(self, alphas, positive=False):
        knees = []
        
        #Set the parameters for the figure 
        num_subplots = len(self.type_names)
        num_rows = int(np.ceil(float(num_subplots)/2))
        plt.figure(figsize=(12, 6 * num_rows))
        
        for i, cell_type in enumerate(self.type_names):
            print('Started cell type: ', cell_type, 'at: ', datetime.now())
            scores = []
            num_genes_arr = []

            for alpha in alphas:
                #print('Started training for alpha', alpha, 'at', datetime.now())
                # Train the model at each alpha
                coeffs, R2, intercept = self.train_type_at_hpf(cell_type, alpha, positive=positive)

                # Score each alpha by it's resultant R^2 value
                scores.append(R2)

                # Also keep track of the number of genes in the gene list
                num_genes = np.count_nonzero(coeffs)
                num_genes_arr.append(num_genes)

            # Plot R^2 vs number of genes
            plt.subplot(num_rows, 2, i+1)
            plt.title('# Genes vs. $R^2$ for ' + cell_type)
            plt.scatter(num_genes_arr, scores)
            plt.xlabel("# Genes")
            plt.ylabel("$R^2$")

            #We need to filter identical x values to prevent a division by zero in knee location
            num_genes_arr_filt, scores_filt, alphas_filt = self.filter_for_repeats(num_genes_arr, scores, alphas)

            #Locate the knee
            if len(scores_filt) >= 2:
                kn = KneeLocator(num_genes_arr_filt, scores_filt, S=1, curve='concave', direction='increasing')
                knee = kn.knee
                knees.append(alphas_filt[num_genes_arr_filt.index(knee)])
                
                #Plot the knee as a red dot
                plt.scatter(knee, scores_filt[num_genes_arr_filt.index(knee)], color='red')
                print('Knee found at ', alphas_filt[num_genes_arr_filt.index(knee)])
            else:
                #If we only have one point or less, set the knee manually
                knees.append(-1)

            for i in range(len(alphas)):
                plt.annotate(alphas[i], (num_genes_arr[i], scores[i]))
            plt.grid()
        
        plt.show()
        return knees