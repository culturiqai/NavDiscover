"""
Viscosity Scaling Analysis for Navier-Stokes Singularities
==========================================================

This script runs a sweep of simulations at a fixed high resolution (128³)
to analyze the scaling of the blow-up time with viscosity (ν). This is a
classic test to distinguish a true singularity from a viscous effect.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# For symbolic regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import sympy as sp

# --- Simulation Parameters ---
N_RUN = 128      # High-resolution grid size for the main runs
N_DISCOVER = 16  # Low-resolution grid for AI discovery
TARGET_SEPARATION = 1.26 # The "sweet spot" separation distance
VISCOSITY_SWEEP = [0.01, 0.005, 0.0025, 0.00125] # Sweep over these ν values
SIM_TIME = 0.4     # Total simulation time
TIME_STEP = 0.0005 # Time step
PLOT_INDIVIDUAL_RUNS = False # Set to True to save plots for each viscosity

# (All classes: HRFNavierStokesPOC_3D, SymbolicRegressionAnalyzer, FilterDiscovery
#  are copied verbatim from clay_3d.py here)
class HRFNavierStokesPOC_3D:
    def __init__(self, N=32, nu=0.01):
        self.N, self.nu = N, nu
        self.discovered_filter = None
        kx_1d, ky_1d, kz_1d = (np.fft.fftfreq(N) * N,)*3
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_nozero = self.k2.copy(); self.k2_nozero[0,0,0] = 1e-10
        kmax = N*2/3/2
        self.dealias_mask = (np.abs(self.kx)<kmax) & (np.abs(self.ky)<kmax) & (np.abs(self.kz)<kmax)
        self.coefficient_history, self.time_history, self.energy_history, self.enstrophy_history, self.singularity_indicators = [],[],[],[],[]
    def simulate(self, u0, v0, w0, T=1.0, dt=0.001, verbose=True):
        u_hat, v_hat, w_hat = fftn(u0), fftn(v0), fftn(w0)
        if self.discovered_filter is not None:
            if verbose: print("Applying discovered spectral filter...")
            u_hat *= self.discovered_filter; v_hat *= self.discovered_filter; w_hat *= self.discovered_filter
        t, step = 0, 0
        if verbose: print(f"Starting {self.N}³ 3D simulation for ν={self.nu:.5f}...")
        while t < T:
            self.coefficient_history.append({'u_hat':u_hat.copy(),'v_hat':v_hat.copy(),'w_hat':w_hat.copy()}); self.time_history.append(t)
            self._calculate_indicators(u_hat, v_hat, w_hat)
            max_coeff = max(np.max(np.abs(u_hat)), np.max(np.abs(v_hat)), np.max(np.abs(w_hat)))
            if max_coeff > 1e12 or np.isnan(max_coeff):
                if verbose: print(f"Potential blow-up detected at t={t:.4f}, max|coeff|={max_coeff:.2e}")
                break
            u_hat, v_hat, w_hat = self._time_step(u_hat, v_hat, w_hat, dt)
            t += dt; step += 1
            if verbose and step % 20 == 0: print(f"  t={t:.4f}, max|u_hat|={np.max(np.abs(u_hat)):.3e}")
        return self._analyze_results()
    def _time_step(self, u_hat, v_hat, w_hat, dt):
        u,v,w = ifftn(u_hat).real, ifftn(v_hat).real, ifftn(w_hat).real
        ux,uy,uz = ifftn(1j*self.kx*u_hat).real, ifftn(1j*self.ky*u_hat).real, ifftn(1j*self.kz*u_hat).real
        vx,vy,vz = ifftn(1j*self.kx*v_hat).real, ifftn(1j*self.ky*v_hat).real, ifftn(1j*self.kz*v_hat).real
        wx,wy,wz = ifftn(1j*self.kx*w_hat).real, ifftn(1j*self.ky*w_hat).real, ifftn(1j*self.kz*w_hat).real
        Nu,Nv,Nw = -(u*ux+v*uy+w*uz), -(u*vx+v*vy+w*vz), -(u*wx+v*wy+w*wz)
        Nu_hat,Nv_hat,Nw_hat = fftn(Nu),fftn(Nv),fftn(Nw)
        Nu_hat[~self.dealias_mask]=0; Nv_hat[~self.dealias_mask]=0; Nw_hat[~self.dealias_mask]=0
        diff_term = 1/(1+self.nu*dt*self.k2)
        u_star,v_star,w_star = (u_hat+dt*Nu_hat)*diff_term, (v_hat+dt*Nv_hat)*diff_term, (w_hat+dt*Nw_hat)*diff_term
        div_u_star = 1j*self.kx*u_star + 1j*self.ky*v_star + 1j*self.kz*w_star
        p_hat = (1j/(dt*self.k2_nozero))*div_u_star
        u_hat_new = u_star - dt*(1j*self.kx*p_hat)
        v_hat_new = v_star - dt*(1j*self.ky*p_hat)
        w_hat_new = w_star - dt*(1j*self.kz*p_hat)
        return u_hat_new, v_hat_new, w_hat_new
    def _calculate_indicators(self,u,v,w):
        energy_density = 0.5*(np.abs(u)**2+np.abs(v)**2+np.abs(w)**2); energy = energy_density/(self.N**6)
        self.energy_history.append(np.sum(energy)); self.enstrophy_history.append(np.sum(self.k2*energy))
        high_k_mask = self.k2 > (self.N/4)**2
        self.singularity_indicators.append(np.sum(energy[high_k_mask])/(np.sum(energy)+1e-12))
    def _analyze_results(self):
        if not self.coefficient_history: return {'time_history':[]}
        final_coeffs = self.coefficient_history[-1]
        return {'coefficient_history':self.coefficient_history, 'time_history':self.time_history, 'energy_history':self.energy_history,
                'enstrophy_history':self.enstrophy_history, 'singularity_indicators':self.singularity_indicators,
                'final_state': {'u_hat':final_coeffs['u_hat'],'v_hat':final_coeffs['v_hat'],'w_hat':final_coeffs['w_hat']}}

class SymbolicRegressionAnalyzer:
    def __init__(self): self.discovered_laws=[]
    def discover_blow_up_law(self, times, max_coeffs):
        if len(times) < 5: return {}
        X, y = times.reshape(-1,1), np.log(max_coeffs+1e-12)
        best_score, best_model, best_formula = -np.inf,None,None
        for degree in range(1,5):
            model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.01)); model.fit(X,y)
            if model.score(X,y) > best_score:
                best_score, best_model = model.score(X,y), model
                t = sp.Symbol('t'); formula = model.named_steps['ridge'].intercept_
                for i,name in enumerate(model.named_steps['polynomialfeatures'].get_feature_names_out(['t'])[1:]):
                    formula += model.named_steps['ridge'].coef_[i+1] * t**(int(name.split('^')[1]) if '^' in name else 1)
                best_formula = sp.exp(formula)
        blow_up_time = self._estimate_blow_up_time(times, max_coeffs)
        return {'formula':best_formula, 'score':best_score, 'blow_up_time':blow_up_time, 'type':self._classify_singularity(times,max_coeffs)}
    def _estimate_blow_up_time(self,times,values):
        if len(times)<5: return None
        log_values=np.log(values+1e-12); slope=(log_values[-1]-log_values[-5])/(times[-1]-times[-5])
        return times[-1]+(25-log_values[-1])/slope if slope>0 else None
    def _classify_singularity(self,times,values):
        if len(values)<10: return "unknown"
        growth_rates=np.diff(np.log(values+1e-12))/np.diff(times)
        if np.mean(np.diff(growth_rates))>0.01: return "super-exponential"
        return "exponential" if np.std(growth_rates)<0.1*np.mean(growth_rates) else "irregular"

class FilterDiscovery:
    def __init__(self, N=16, nu=0.01):
        self.N, self.nu = N, nu; self.discovered_filters=[]
        kx_1d,ky_1d,kz_1d=(np.fft.fftfreq(N)*N,)*3
        self.kx,self.ky,self.kz=np.meshgrid(kx_1d,ky_1d,kz_1d,indexing='ij')
        self.k_mag=np.sqrt(self.kx**2+self.ky**2+self.kz**2)
    def discover_filter(self,u0,v0,w0,objective_type='peak_enstrophy'):
        print(f"\n-- Starting {self.N}³ discovery for ν={self.nu:.5f} (Objective: {objective_type}) --")
        def objective(params): return -self._evaluate_filter(self._params_to_filter(params),u0,v0,w0,objective_type)
        bounds=[(self.N/8,self.N/2),(self.N/16,self.N/4),(1.0,1.5)]
        result=differential_evolution(objective,bounds,maxiter=10,popsize=8,disp=False)
        optimal_filter=self._params_to_filter(result.x); self.discovered_filters.append(result.x)
        print("-- Filter discovery complete --")
        return optimal_filter, result.x
    def _params_to_filter(self,params):
        center_k,width,amplitude=params
        return 1.0+(amplitude-1.0)*np.exp(-((self.k_mag-center_k)**2)/(2*width**2))
    def _evaluate_filter(self,filter_3d,u0,v0,w0,objective_type):
        temp_solver=HRFNavierStokesPOC_3D(N=self.N,nu=self.nu); temp_solver.discovered_filter=filter_3d
        T_eval=0.1; results=temp_solver.simulate(u0,v0,w0,T=T_eval,dt=0.002,verbose=False)
        if not results['enstrophy_history']: return 0
        if objective_type=='peak_enstrophy': return max(results['enstrophy_history'])
        else: raise ValueError(f"Unknown objective type: {objective_type}")

def create_antiparallel_vortices(N, strength=5.0, separation=np.pi):
    print(f"Creating {N}³ anti-parallel vortices (separation={separation:.2f})...")
    x,y,z=(np.linspace(0,2*np.pi,N,endpoint=False),)*3; X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    u0,v0,w0 = np.zeros((N,N,N)), np.zeros((N,N,N)), np.zeros((N,N,N))
    radius=np.pi/4; x1,y1 = np.pi-separation/2,np.pi/2; x2,y2 = np.pi+separation/2,np.pi/2
    w0 += strength*np.exp(-(((X-x1)**2+(Y-y1)**2)/radius**2))
    w0 -= strength*np.exp(-(((X-x2)**2+(Y-y2)**2)/radius**2))
    noise=0.1; u0+=noise*np.sin(2*Z); v0+=noise*np.cos(2*Z)
    return u0,v0,w0

def run_single_experiment(viscosity):
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {N_RUN}³ with Viscosity ν = {viscosity:.5f}")
    print("="*80)
    
    # 1. AI-Guided Discovery on Low-Resolution Grid
    u0_discover,v0_discover,w0_discover = create_antiparallel_vortices(N_DISCOVER, separation=TARGET_SEPARATION)
    discoverer = FilterDiscovery(N=N_DISCOVER, nu=viscosity)
    discovered_filter_lowres, filter_params = discoverer.discover_filter(u0_discover,v0_discover,w0_discover)
    
    # 2. Set up and Run High-Resolution Simulation
    u0_hires,v0_hires,w0_hires = create_antiparallel_vortices(N_RUN, separation=TARGET_SEPARATION)
    
    solver = HRFNavierStokesPOC_3D(N=N_RUN, nu=viscosity)
    
    # Upscale the discovered filter
    scale_factor = N_RUN // N_DISCOVER
    solver.discovered_filter = np.kron(discovered_filter_lowres, np.ones((scale_factor, scale_factor, scale_factor)))
    
    results = solver.simulate(u0_hires, v0_hires, w0_hires, T=SIM_TIME, dt=TIME_STEP)
    
    if len(results['time_history']) < 5:
        print("Simulation finished too quickly. No analysis performed.")
        return None

    # 3. Analyze and Save Results
    times = np.array(results['time_history'])
    max_coeffs = np.array([max(np.max(np.abs(c['u_hat'])),np.max(np.abs(c['v_hat'])),np.max(np.abs(c['w_hat']))) for c in results['coefficient_history']])
    sr_analyzer = SymbolicRegressionAnalyzer()
    blow_up_law = sr_analyzer.discover_blow_up_law(times, max_coeffs)
    
    if PLOT_INDIVIDUAL_RUNS:
        plot_filename = f"hypo_viscosity_nu_{viscosity:.5f}"
        plot_results(times, max_coeffs, results, blow_up_law, f"{plot_filename}_summary.png", viscosity)
        if results.get('final_state'):
            plot_vorticity_slices(results['final_state'], solver, f"{plot_filename}_vorticity.png", viscosity)
    
    return blow_up_law.get('blow_up_time')

def plot_sweep_results(viscosities, blow_up_times):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Viscosity Scaling of Blow-up Time", fontsize=16)
    
    ax.plot(viscosities, blow_up_times, 'o-', label='Simulation Data')
    ax.set_xlabel("Viscosity (ν)")
    ax.set_ylabel("Estimated Blow-up Time (T*)")
    ax.set_title(f"Blow-up Time vs. Viscosity at {N_RUN}³")
    ax.grid(True)
    ax.set_xscale('log') # Log scale for viscosity is often useful
    ax.legend()

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("hypo_viscosity_sweep_summary.png", dpi=150)
    plt.show()

def plot_results(times, max_coeffs, results, blow_up_law, filename, nu):
    fig,axes=plt.subplots(2,2,figsize=(12,10));fig.suptitle(f"Singularity Analysis ({N_RUN}³, ν={nu:.5f})",fontsize=16)
    # ... (plotting code same as before, omitted for brevity) ...

def plot_vorticity_slices(final_state, solver, filename, nu):
    # ... (plotting code same as before, omitted for brevity) ...
    pass

if __name__ == "__main__":
    sweep_results = []
    valid_viscosities = []
    
    for nu in VISCOSITY_SWEEP:
        blow_up_time = run_single_experiment(nu)
        if blow_up_time is not None:
            sweep_results.append(blow_up_time)
            valid_viscosities.append(nu)

    if len(sweep_results) > 1:
        plot_sweep_results(valid_viscosities, sweep_results)

    print("\n\nViscosity sweep complete.")
    print("Check hypo_viscosity_sweep_summary.png for results.") 
