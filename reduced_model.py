import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

save_at = '/Users/avazquez/submissions/membrane_constraint.2024'

class ReducedModel:
    """A class to explore solutions to the reduced model."""

    def __init__(self, sigma_E=0, atp_yield_E = 2):
        
        self.sigma_E = sigma_E
        self.atp_yield_E = atp_yield_E
        
        # references
        
        self.references = {
            'Jeckelmann 2011': '1. Structure and function of the glucose PTS transporter from Escherichia coli. Journal of Structural Biology 176, 395–403 (2011).',
            'Nagle 2000': 'Nagle, J.F. and Tristram-Nagle, S.Structure of lipid bilayers. Biochimica et Biophysica Acta (BBA) - Reviews on Biomembranes 1469, 159–195 (2000).',
            'Rohwer 2000': 'Rohwer, J.M., Meadow, D.N., Roseman, S., Westerhoff, H.V. & Postma, P.W. Understanding Glucose Transport by the Bacterial Phosphoenolpyruvate:Glycose Phosphotransferase System on the Basis of Kinetic Measurements in Vitro. Journal of Biological Chemistry 275, 34909–34921 (2000).',
        }
        
        # parameters

        p0 = dict()
        enzyme = dict()
        pathway = dict()
        
        # basic paramaters
        
        p0['mw_a'] = dict(value=118.9, units='g/mol', name='average molecular weight of amino acid in protein', ref='Bionumbers 107786')
        p0['mw_n'] = dict(value=324.3, units='g/mol', name='average molecular weight of nucleotide in RNA', ref='Bionumbers 104886')
        p0['P'] = dict(value=55, units='% of cell dry weight', name='protein content', ref='Bionumbers 101436')
        p0['N'] = dict(value=20.5, units='% of cell dry weight', name='RNA nucleotide content', ref='Bionumbers 101436')
        p0['nu'] = dict(value=(p0['N']['value'] / p0['mw_n']['value'])/(p0['P']['value'] / p0['mw_a']['value']), units='nucleotide/amino acid', name='RNA nucleotides per amino acid', ref='calculated')
        p0['l'] = dict(value=0.15 * 10000, units='A', name='cell volume to area ratio', ref='Bionumbers 106614')
        p0['a_L'] = dict(value=58, units='A^2', name='', ref='Nagle 2000', ref_note='A in Table 6')
        p0['a_P'] = dict(value=7, units='A^2/amino acid', name='protein membrane area per amino acid', ref='Vazquez & Gedeon 2024')
        p0['l_P'] = dict(value=90, units='A', name='protein volume to area ratio', ref='Vazquez & Gedeon 2024')

        # pathway parameters
        
        param = dict()
        
        param['e_P'] = dict(value=4.2, min=4.2, max=4.2, ref='Bionumbers 114971')
        param['e_N'] = dict(value=18, min=18, max=18, ref='Vazquez & Gedeon 2024')
        param['e_L'] = dict(value=4, min=4, max=4, ref='Vazquez & Gedeon 2024')
        param['nu'] = dict(value=p0['nu']['value'], min=p0['nu']['value'], max=p0['nu']['value'])

        param['sigma_P'] = dict(value=0, min=0, max=0)
        param['sigma_N'] = dict(value=0, min=0, max=0)
        param['sigma_L'] = dict(value=1, min=1, max=1)
        param['sigma_E'] = dict(value=sigma_E, min=sigma_E, max=sigma_E)  # glycolysis

        # effective rates (Vazquez & Gedeon 2024)
        lambda_P = 0.0028
        lambda_N = 0.026
        lambda_L = 0.03
        lambda_E = 0.01 * atp_yield_E
        lambda_PTS = 0.72
        uncertainty = 2
        param['lambda_P'] = dict(value=lambda_P, min=lambda_P/uncertainty, max=lambda_P*uncertainty)
        param['lambda_N'] = dict(value=lambda_N, min=lambda_N/uncertainty, max=lambda_N*uncertainty)
        param['lambda_L'] = dict(value=lambda_L, min=lambda_L/uncertainty, max=lambda_L*uncertainty)
        param['lambda_E'] = dict(value=lambda_E, min=lambda_E/uncertainty, max=lambda_E*uncertainty)
        param['lambda_glc'] = dict(value=lambda_PTS, min=lambda_PTS/uncertainty, max=lambda_PTS*2)
        
        param['K_glc'] = dict(value=20, min=20, max=20, ref='Rohwer 2000')
 
        # geometric parameters
    
        param['phi'] = dict(value=(0.34+0.44)/2, min=0.34, max=0.44, name='macromolecular volume fraction', ref='Bionumbers 105814')
        a_L_over_a_P = p0['a_L']['value'] / p0['a_P']['value']
        l_P_over_l = p0['l_P']['value'] / p0['l']['value']
        param['a_L_over_a_P'] = dict(value=a_L_over_a_P, min=a_L_over_a_P/2, max=a_L_over_a_P*2)
        param['l_P_over_l'] = dict(value=l_P_over_l, min=l_P_over_l/2, max=l_P_over_l*2)
        
        # parameters with uncertainty
        for k in param.keys():
            param[k]['is_uncertain'] = param[k]['min'] < param[k]['max']
        
        self.param_0 = p0
        self.enzyme = enzyme
        self.pathway = pathway
        self.param = param
        self.param_default = self.param.copy()
        
        self.x = None
        
    def restore_default_param(self):
        self.param = self.param_default.copy()

    def create_scenario(self, n_realisations=1, key=None, start=None, end=None, step=None, glc_over_K=None, change_parameters={}):
        
        for k in change_parameters.keys():
            if k in self.param :
                self.param[k]['value'] = change_parameters[k]['value']
                self.param[k]['min'] = change_parameters[k]['min']
                self.param[k]['max'] = change_parameters[k]['max']
                self.param[k]['is_uncertain'] = self.param[k]['min'] < self.param[k]['max']
            else:
                raise ValueError(f'{k} is not a model parameter') 

        v = dict()
        for k in self.param.keys():
            v[k] = np.random.triangular(self.param[k]['min'], self.param[k]['value'], self.param[k]['max'], n_realisations) if self.param[k]['is_uncertain'] else np.repeat(self.param[k]['value'], n_realisations)
        
        if key is not None:
            self.x = np.arange(start, end, step)
            n_points = self.x.size
            self.x = np.repeat(self.x[np.newaxis, :], n_realisations, axis=0)
            for k in self.param.keys():
                v[k] = np.repeat(v[k][:, np.newaxis], n_points, axis=1)
            if key in self.param:
                v[key] = self.x
   
        if key == 'glc_over_K':
            lambda_inv_pts = v['K_glc'] / (v['lambda_glc'] * self.x)
            v['lambda_E'] = self.atp_yield_E /(self.atp_yield_E/v['lambda_E'] + lambda_inv_pts)
            for p in ['P', 'N', 'L']:
                v[f'lambda_{p}'] = 1/(1/v[f'lambda_{p}'] + lambda_inv_pts)
            for p in ['P', 'N', 'L', 'E']:
                v[f'sigma_{p}'] = ( v[f'sigma_{p}'] / v[f'lambda_{p}'] + lambda_inv_pts) * v[f'lambda_{p}']

        # calculations
        alpha_P = 1 + v['e_P'] * v['lambda_P'] / v['lambda_E']
        alpha_N = 1 + v['e_N'] * v['lambda_N'] / v['lambda_E']
        alpha_L = 1 + v['e_L'] * v['lambda_L'] / v['lambda_E']
        alpha_PN = alpha_P + (v['nu'] * v['lambda_P'] / v['lambda_N']) * alpha_N
        beta_P = v['sigma_P'] + v['sigma_E'] * v['e_P'] * v['lambda_P'] / v['lambda_E']
        beta_N = v['sigma_N'] + v['sigma_E'] * v['e_N'] * v['lambda_N'] / v['lambda_E']
        beta_L = v['sigma_L'] + v['sigma_E'] * v['e_L'] * v['lambda_L'] / v['lambda_E']
        beta_PN = beta_P + (v['nu'] * v['lambda_P'] / v['lambda_N']) * beta_N
        self.theta = v['a_L_over_a_P'] * v['lambda_L'] / v['lambda_P']
        self.pi_bar_A = v['l_P_over_l'] / v['phi']
        self.pi_star_A = self.pi_bar_A + (self.theta * alpha_PN / alpha_L) - (beta_L / alpha_L)
        self.kappa = beta_PN - (beta_L * alpha_PN / alpha_L)
        self.Delta = self.pi_star_A**2 - (4 * self.kappa * self.theta / alpha_L)
    
        # solution to the quadratic equation

        self.pi_P = ( self.pi_star_A - np.sqrt(self.Delta) ) / (2*self.kappa)
    
        self.pi_L = (1 - alpha_PN * self.pi_P) / alpha_L
        self.pi_N = self.pi_P * v['nu'] * v['lambda_P'] / v['lambda_N']
        self.pi_E = (alpha_P-1) * self.pi_P + (alpha_N-1) * self.pi_N + (alpha_L-1) * self.pi_L
        self.pi_A = v['sigma_P'] * self.pi_P + v['sigma_N'] * self.pi_N + v['sigma_L'] * self.pi_L + v['sigma_E'] * self.pi_E
    
        self.A_P = self.pi_A / (v['a_L_over_a_P'] * (v['lambda_L'] * self.pi_L) / (v['lambda_P'] * self.pi_P) + self.pi_A )
    
        # without membrane
        self.pi_P_0 = 1 / alpha_PN
    
    def plot_1(self, key, xlabel, name, xtra=None, plot_pi_L_0_line=False, plot_dense_packing_line=False, plot_dense_packing_line_legend='upper right'):
        
        if self.x is None:
            raise ValueError('create scenario to plot results')

        x = self.x[0]
        pi_P = np.quantile(self.pi_P, [0.2, 0.5, 0.8], axis=0)
        pi_N = np.quantile(self.pi_N, [0.2, 0.5, 0.8], axis=0)
        pi_L = np.quantile(self.pi_L, [0.2, 0.5, 0.8], axis=0)
        pi_E = np.quantile(self.pi_E, [0.2, 0.5, 0.8], axis=0)
        pi_A = np.quantile(self.pi_A, [0.2, 0.5, 0.8], axis=0)
        A_P = np.quantile(self.A_P, [0.2, 0.5, 0.8], axis=0)
        kappa = np.quantile(self.kappa, [0.2, 0.5, 0.8], axis=0)

        fig, ax = plt.subplots(3,2)

        st = dict(alpha=0.5)
        stl = dict(lw=2,color='black')

        i = 0
        j = 0
        t = 'A)'
        ax[i,j].fill_between(x, kappa[0, :], kappa[2, :], **st)
        ax[i,j].plot(x, kappa[1, :], **stl)
        ax[i,j].set(ylabel=r'$\kappa$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            ax[i,j].set_xticklabels([])
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.set(xlabel=xtra)
            ax2.plot(scale * x, kappa[1, :], **stl)

        i = 0
        j = 1
        t = 'B)'
        ax[i,j].fill_between(x, pi_P[0, :], pi_P[2, :], **st)
        ax[i,j].plot(x, pi_P[1, :], **stl)
        ax[i,j].set(ylabel='$\pi_P$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            ax[i,j].set_xticklabels([])
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.set(xlabel=xtra)
            ax2.plot(scale * x, pi_P[1, :], **stl)

        i = 1
        j = 0
        t = 'C)'
        ax[i,j].fill_between(x, pi_N[0, :], pi_N[2, :], **st)
        ax[i,j].plot(x, pi_N[1, :], **stl)
        ax[i,j].set(ylabel='$\pi_N$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            ax[i,j].set_xticklabels([])
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.plot(scale * x, pi_N[1, :], **stl)
            ax2.set_xticklabels([])

        i = 1
        j = 1
        t = 'D)'
        ax[i,j].fill_between(x, pi_L[0, :], pi_L[2, :], **st)
        ax[i,j].plot(x, pi_L[1, :], **stl)
        if plot_pi_L_0_line:
            ax[i,j].plot([x.min(), x.max()], [0,0], 'k--')
        ax[i,j].set(ylabel='$\pi_L$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            ax[i,j].set_xticklabels([])
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.plot(scale * x, pi_L[1, :], **stl)
            ax2.set_xticklabels([])

        i = 2
        j = 0
        t = 'E)'
        ax[i,j].fill_between(x, pi_E[0, :], pi_E[2, :], **st)
        ax[i,j].plot(x, pi_E[1, :], **stl)
        ax[i,j].set(xlabel=xlabel, ylabel=r'$\pi_E$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.plot(scale * x, pi_E[1, :], **stl)
            ax2.set_xticklabels([])
    
        i = 2
        j = 1
        t = 'F)'
        #ax[i,j].fill_between(x, pi_A[0, :], pi_A[2, :], **st)
        #ax[i,j].plot(x, pi_A[1, :], **stl)
        #ax[i,j].set(xlabel=xlabel, ylabel=r'$\pi_A$')
        ax[i,j].fill_between(x, A_P[0, :], A_P[2, :], **st)
        ax[i,j].plot(x, A_P[1, :], **stl)
        if plot_dense_packing_line:
            ax[i,j].plot([x.min(), x.max()], [1,1], 'k--', label='densest disk packing')
            #ax[i,j].plot([x.min(), x.max()], [0.91,0.91], 'k--', label='densest disk packing')
            #ax[i,j].legend(loc=plot_dense_packing_line_legend, frameon = 0, handletextpad = 0, fontsize = 14)
        ax[i,j].set(xlabel=xlabel, ylabel=r'$A_P$')
        ax[i,j].set_title(t, x=-0.1, y=1.05)
        if xtra is not None:
            if xtra == r'$l$ ($\mu$m)':
                scale = 0.0001 * self.param_0['l']['value'] / self.param['phi']['value']
            ax2 = ax[i,j].twiny()
            ax2.plot(scale * x, A_P[1, :], **stl)
            ax2.set_xticklabels([])

        plt.subplots_adjust(bottom=0, right=2, wspace=0.3, top=2, hspace=0.3)

        plt.savefig(f'{save_at}/{name}.pdf',bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)


model = ReducedModel()

m = 10000
model.create_scenario(n_realisations=m)

pi_P = model.pi_P
pi_P_0 = model.pi_P_0
print(pi_P.mean(), pi_P_0.mean())

fig, ax = plt.subplots()
hist, bins = np.histogram(pi_P, bins=20)
ax.vlines(0.5*(bins[:-1]+bins[1:]), 0, hist/m, lw=10)
ax.set(xlabel=r'$\pi_P$', ylabel='Frequency')
#fig, ax = plt.subplots(1,2)
#ax[0].vlines(0.5*(bins[:-1]+bins[1:]), 0, hist/m, lw=10)
#ax[0].set(xlabel=r'$\pi_P$', ylabel='Frequency')
#ax[0].set_title('A)  With volume-area balance', x=0.1, y=1.05)
#hist, bins = np.histogram(pi_P_0, bins=20)
#ax[1].vlines(0.5*(bins[:-1]+bins[1:]), 0, hist/m, lw=10)
#ax[1].set(xlabel=r'$\pi_P$', ylabel='Frequency')
#ax[1].set_title('B)  Without volume-area balance', x=0.1, y=1.05)
#plt.subplots_adjust(bottom=0, right=2, wspace=0.3)
plt.savefig(f'{save_at}/pi_P.pdf',bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)

print('pi_P_phi')
model.create_scenario(n_realisations=10000, key='phi', start=0.00001, end=0.4, step=0.001)
model.plot_1(key='phi', xlabel=r'$\phi$', name='pi_P_phi', xtra=r'$l$ ($\mu$m)')

model.create_scenario(n_realisations=10000, key='glc_over_K', start=2, end=50, step=1)
model.plot_1(key='glc_over_K', xlabel= r'[Glucose] ($\mu$M)', name='pi_P_glc', plot_pi_L_0_line=True, plot_dense_packing_line=True)

model.create_scenario(n_realisations=10000, key='glc_over_K', start=2, end=50, step=0.1, change_parameters=dict(phi=dict(value=0.1, min=0.1, max=0.1)))
model.plot_1(key='glc_over_K', xlabel= r'${\rm [Glucose]}/K_{\rm Glucose}$', name='pi_P_glc_phi=0.1', plot_pi_L_0_line=True)

# oxphos

print('pi_P_phi_oxphos')
model = ReducedModel(sigma_E=1, atp_yield_E=26)
model.create_scenario(n_realisations=10000, key='phi', start=0.00001, end=0.4, step=0.001)
model.plot_1(key='phi', xlabel=r'$\phi$', name='pi_P_phi_oxphos', xtra=r'$l$ ($\mu$m)')

print('pi_P_glc_oxphos')
model.create_scenario(n_realisations=10000, key='glc_over_K', start=2, end=50, step=1)
model.plot_1(key='glc_over_K', xlabel= r'[Glucose] ($\mu$M)', name='pi_P_glc_oxphos', plot_pi_L_0_line=True, plot_dense_packing_line=True)

