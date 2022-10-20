import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import anndata
from io import BytesIO
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from kneed import KneeLocator

class Fate_Lasso():
    
    # An object that can perform Lasso regressions on gene expression
    # vs. fate.
    #
    # INPUT:
    # adata_full: An anndata containing all the cells given in the fates dataframe.
    # fates: A pandas df containing cells on the rows and cell types.
    #        on the columns. Entries are fate probabilities for each cell type.
    # type_names: The type_names that are of interest for the regressions.
    # factor_genes: A list of genes which we'll restrict the regressions to.
    # common_names: A pandas df translating gene names to common names.
    # indices: The indices of the cells we will consider.
    def __init__(self, adata_full, fates, type_names,
                 factor_genes, common_names, indices, train_ratio):
        self.adata_full = adata_full
        
        #Copy the fates so we don't modify the original
        fates_c = fates.copy()
        self.fates = fates_c
        
        self.type_names = type_names
        self.factor_genes = factor_genes
        self.common_names = common_names
        
        self.TRAIN_RATIO = train_ratio
        self.train, self.test = train_test_split(indices, train_size=self.TRAIN_RATIO)
        
        #Memoize previously computed fits
        self.computed_fits = {cell_type: {} for cell_type in type_names}
        
    #################################################
    # Core regression code
    #################################################
    
    # Fits the cell fates for the given celltype using the Lasso model.
    # The gene expression data on the "factor" genes is used as the independant
    # variable.
    # INPUT
    # cell_type: The celltype whose fate we want to fit
    # alpha: The alpha parameter for the Lasso model which controls the sparsity
    # positive: Whether to fit the Lasso using positive coefficients
    #
    # OUTPUT
    # coeffs: A vector of regression coefficients whose length is the number of
    #         genes we used
    # intercept: The intercept term for the regression
    # R2: The R^2 value of the fit
    def _train_celltype(self, cell_type, alpha, positive=False):    
        #Get training expression data and bary coords
        train_exp_data = self.adata_full[self.train, self.factor_genes].X
        train_bary_data = self.fates.loc[self.train, cell_type]
        train_bary_data = train_bary_data.to_numpy()
        
        #Train the model, checking if we already computed for this alpha
        if str(alpha) in self.computed_fits[cell_type]:
            reg = self.computed_fits[cell_type][str(alpha)]
        else:
            reg = linear_model.Lasso(alpha=alpha, positive=positive, random_state=0)
            reg.fit(train_exp_data, train_bary_data)
            
            self.computed_fits[cell_type][str(alpha)] = reg
      
        coeffs = reg.coef_
        intercept = reg.intercept_
        
        #Get R^2 score
        R2 = reg.score(self.adata_full[self.test, self.factor_genes].X, self.fates.loc[self.test, cell_type])

        return coeffs, intercept, R2
    
    # Given a list of cells, trains against all possible cell types.
    # Returns the found coefficients arranged by cell type, the R^2
    # value by cell type and the intercepts for the fits.
    # 
    # INPUT:
    # alphas: A list of alphas corresponding to celltypes. Each celltype is
    #         trained for its corresponding alpha.
    #
    # OUTPUT:
    # coeffs_arr: A dict of coefficients where coeffs_arr[celltype] is a list of coeffs
    #             for that celltype.
    # R2_arr: A dict of R^2 values for each celltype.
    # intercepts: A dict of intercepts for each celltype.
    def _train_cells(self, alphas, positive=False):
        #Store the results of each fit
        coeffs_arr = {cell_type: [] for cell_type in self.type_names}
        R2_arr = {cell_type: -9999 for cell_type in self.type_names}
        intercepts = {cell_type: -9999 for cell_type in self.type_names}
        
        for cell_type, alpha in zip(self.type_names, alphas):
            coeffs, intercept, R2 = self._train_celltype(cell_type, alpha, positive=positive)
            coeffs_arr[cell_type] = coeffs
            R2_arr[cell_type] = R2
            intercepts[cell_type] = intercept
            print('Finished training type =  ', cell_type)

        return coeffs_arr, R2_arr, intercepts
    
    # Given a list of fit coefficients for each cell type,
    # an array of R^2 values for each cell type, and intercepts
    # puts everything in a convenient dict by celltype along with
    # the names of the non-zero genes.
    #
    # INPUT
    # coeffs_arr: A dict of coefficient lists for each celltype.
    # R2_arr: A dict of R^2 values for each celltype.
    # intercepts: A dict of intercepts for each celltype.
    #
    # OUTPUT
    # genes_list: A dict with entries for each celltype containing non-zero genes,
    #             their coefficients, the R^2 value, and the intercept.
    def _make_gene_lists(self, coeffs_arr, R2_arr, intercepts):
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
    
    # Given the indices of test cells and an array of R^2 values
    # prints metrics about the quality of fit. These currently include
    # avg. residual, std. dev. of residuals, R^2.
    #
    # INPUT
    # R2_arr: A dict of R^2 values by celltype
    def _analyze_fit(self, R2_arr):
        #Go through each cell type and calculate the residual
        for cell_type in self.type_names:
            score = R2_arr[cell_type]
            print('R^2 for {} is:'.format(cell_type), score)
    
    # Trains the each celltype at its corresponding alpha and
    # prints some summary statistics.
    #
    # INPUT:
    # alphas: The list of alphas corresponding to the celltypes.
    # positive: Whether to use only positive coefficients for the fit.
    #
    # OUTPUT:
    # genes_list: A dict with entries for each celltype containing non-zero genes,
    #             their coefficients, the R^2 value, and the intercept.
    def train_analyze(self, alphas, positive=False):
        #Run training
        coeffs_arr, R2_arr, intercepts = self._train_cells(alphas, positive=positive)
        print()
        #Make gene lists
        genes_list = self._make_gene_lists(coeffs_arr, R2_arr, intercepts)
        print()
        #Print stats about the fit
        self._analyze_fit(R2_arr)
        return genes_list
    
    #Given a set of gene_lists computed at mulitple alphas,
    #and a pandas df of ATG names to common names,
    #converts the ATG names to common.
    #
    # INPUT
    # gene_list: A gene list, formatted like in train_analyze.
    # common_names: A pd df translating gene names to common names.
    #
    # OUTPUT
    # gene_list: The same gene "list" but with the list of genes translated to common
    #            names, and the AT names added on as a separate list.
    def list_to_common_names(self, gene_list, common_names):
        #Go through each alpha and change genes for each cell type
        for cell_type in self.type_names:
            ATG_names = gene_list[cell_type]['genes']
            common = list(self.common_names.loc[ATG_names].Name)
            gene_list[cell_type]['genes'] = common
            gene_list[cell_type]['Gene AT Identifier Number'] = ATG_names

        return gene_list
    
    ##########################################################
    # Plotting code that can be used to determine default
    # values for alpha
    ##########################################################
    
    # Given a list of numbers of genes, scores, and alphas, filters
    # the number of genes list for repeat values that occur next to each other and removes
    # those indices from the other two lists. 
    #
    # INPUT
    # num_genes_arr: A list containing numbers of non-zero genes from the regression.
    # scores: A list of R^2 scores. Must be the same length as num_genes_arr.
    # alphas: A list of alphas for which we fit the Lasso. Must be the same length as num_genes_arr.
    #
    # OUTPUT
    # num_genes_arr_filt: num_genes_arr, but with the first instance for each repeated value removed.
    # scores_filt: scores, but with the same indices removed as from num_genes_arr.
    # alphas_filt: alphas, but with the same indices removed as from num_genes_arr.
    def _filter_for_repeats(self, num_genes_arr, scores, alphas):
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
    
    # Given cells to fit on and alphas, creates plots
    # of R^2 vs. Number of genes with non-zero coeffs at each alpha.
    # This is done for each cell type. The knee point is marked on the plots
    # and returned.
    #
    # INPUT
    # alphas: A list of alphas for the Lasso regression.
    # positive: Whether to fit only with positive coefficients for Lasso.
    # save_path: If specified, saves the figure to the given path
    # annotate: Takes values "all" or "knee". 
    #           Whether to label all points on the plot, or just the knee.
    #
    # OUTPUT
    # knees: The knee points for each celltype.
    def make_plots_R2_num_genes(self, 
                                alphas, 
                                positive=False, 
                                save_path=None,
                                annotate='knee'):
        knees = []
        
        #Set the parameters for the figure 
        num_subplots = len(self.type_names)
        num_rows = int(np.ceil(float(num_subplots)/2))
        plt.figure(figsize=(2*7.2/5 * num_rows, 2*3.5))
        
        for i, cell_type in enumerate(self.type_names):
            print('Started cell type: ', cell_type, 'at: ', datetime.now())
            scores = []
            num_genes_arr = []

            for alpha in alphas:
                # Train the model at each alpha
                coeffs, intercept, R2 = self._train_celltype(cell_type, alpha, positive=positive)

                # Score each alpha by it's resultant R^2 value
                scores.append(R2)

                # Also keep track of the number of genes in the gene list
                num_genes = np.count_nonzero(coeffs)
                num_genes_arr.append(num_genes)

            # Plot R^2 vs number of genes
            plt.subplot(2, num_rows, i+1)
            plt.title(cell_type, fontsize=12)
            plt.scatter(num_genes_arr, scores)
            plt.xlabel("Number of Genes", fontsize=12)
            plt.ylabel("$R^2$", fontsize=12)

            #We need to filter identical x values to prevent a division by zero in knee location
            num_genes_arr_filt, scores_filt, alphas_filt = self._filter_for_repeats(num_genes_arr, scores, alphas)

            #Locate the knee
            if len(scores_filt) >= 2:
                kn = KneeLocator(num_genes_arr_filt, scores_filt, S=1, curve='concave', direction='increasing')
                knee = kn.knee
                
                alpha_knee = alphas_filt[num_genes_arr_filt.index(knee)]
                scores_knee = scores_filt[num_genes_arr_filt.index(knee)]
                knees.append(alpha_knee)
                
                #Plot the knee as a red dot
                plt.scatter(knee, scores_knee, color='red')
                print('Knee found at ', alphas_filt[num_genes_arr_filt.index(knee)])
            else:
                #If we only have one point or less, set the knee manually
                alpha_knee = None
                knees.append(-1)
            
            if annotate == 'all':
                for i in range(len(alphas)):
                    plt.annotate(alphas[i], (num_genes_arr[i], scores[i]))
            if (annotate == 'knee') and (alpha_knee is not None):
                plt.annotate(alpha_knee, (knee + 5, scores_knee-0.05), fontsize=12)
                plt.ylim(0, 0.9)
                plt.xlim(0, 300)
            
            plt.grid()
        
        plt.tight_layout()
        
        # Optionally save the figure
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
        return knees