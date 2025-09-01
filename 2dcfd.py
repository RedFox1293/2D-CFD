import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class SimpleCFDSolver:
    def __init__(self, nx=60, ny=30, cylinder_radius=0.5, domain_length=8, domain_height=4):
        # Domain parameters
        self.L = domain_length
        self.H = domain_height
        self.cylinder_radius = cylinder_radius
        self.cylinder_center = (2.0, self.H/2)
        
        # Create stretched grid
        self.nx = nx
        self.ny = ny
        self.create_stretched_grid()
        
        # Flow parameters
        self.rho = 1.0
        self.mu = 0.05  # Higher viscosity for stability
        self.inlet_velocity = 1.0
        self.dt = 0.02  # Larger time step
        
        # Initialize fields
        self.u = np.ones((self.ny, self.nx)) * self.inlet_velocity
        self.v = np.zeros((self.ny, self.nx))
        self.p = np.zeros((self.ny, self.nx))
        
        # Spalart-Allmaras variable (simplified)
        self.nu_tilde = np.zeros((self.ny, self.nx))
        
        # Solver parameters
        self.max_iter = 2000
        self.tolerance = 5e-2  # Relaxed tolerance
        self.omega = 0.5  # Fixed moderate relaxation
        self.residuals = []
        self.iteration = 0
        self.solving = False
        
        # Apply initial conditions
        self.initialize_flow()
        
    def create_stretched_grid(self):
        """Create stretched Cartesian grid with refinement near cylinder"""
        cx, cy = self.cylinder_center
        
        # X-direction: refine near cylinder
        x_uniform = np.linspace(0, self.L, self.nx//2)
        x_refined = []
        
        for x in x_uniform:
            x_refined.append(x)
            # Add extra points near cylinder
            if abs(x - cx) < 2.0:
                if len(x_refined) > 1:
                    x_mid = (x_refined[-2] + x_refined[-1]) / 2
                    x_refined.insert(-1, x_mid)
        
        self.x = np.array(sorted(set(x_refined)))[:self.nx]
        if len(self.x) < self.nx:
            self.x = np.linspace(0, self.L, self.nx)
        
        # Y-direction: refine near centerline
        y_uniform = np.linspace(0, self.H, self.ny//2)
        y_refined = []
        
        for y in y_uniform:
            y_refined.append(y)
            if abs(y - cy) < 1.0:
                if len(y_refined) > 1:
                    y_mid = (y_refined[-2] + y_refined[-1]) / 2
                    y_refined.insert(-1, y_mid)
        
        self.y = np.array(sorted(set(y_refined)))[:self.ny]
        if len(self.y) < self.ny:
            self.y = np.linspace(0, self.H, self.ny)
        
        # Create mesh
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = np.mean(np.diff(self.x))
        self.dy = np.mean(np.diff(self.y))
        
        # Mark solid cells (cylinder)
        self.solid = np.zeros((self.ny, self.nx), dtype=bool)
        for i in range(self.ny):
            for j in range(self.nx):
                dist = np.sqrt((self.X[i,j] - cx)**2 + (self.Y[i,j] - cy)**2)
                if dist <= self.cylinder_radius:
                    self.solid[i,j] = True
        
        print(f"Grid: {self.nx}×{self.ny}, dx≈{self.dx:.3f}, dy≈{self.dy:.3f}")
    
    def initialize_flow(self):
        """Initialize flow field"""
        # Uniform inlet with smooth startup
        for j in range(self.nx):
            if self.x[j] < 1.0:
                factor = self.x[j]
            else:
                factor = 1.0
            self.u[:, j] = self.inlet_velocity * factor
        
        # Zero velocity in cylinder
        self.u[self.solid] = 0
        self.v[self.solid] = 0
        
        # Small perturbation
        self.v += 0.01 * np.random.randn(self.ny, self.nx)
        self.v[self.solid] = 0
    
    def apply_bc(self):
        """Apply boundary conditions"""
        # Inlet
        self.u[:, 0] = self.inlet_velocity
        self.v[:, 0] = 0
        
        # Outlet
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        self.p[:, -1] = 0
        
        # Walls
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0
        
        # Cylinder
        self.u[self.solid] = 0
        self.v[self.solid] = 0
    
    def solve_spalart_allmaras_simple(self):
        """Simplified SA turbulence model"""
        # Simple mixing length model instead of full SA
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if not self.solid[i,j]:
                    # Distance to nearest wall
                    dist = min(i * self.dy, (self.ny - 1 - i) * self.dy)
                    
                    # Check distance to cylinder
                    cx, cy = self.cylinder_center
                    dist_cyl = np.sqrt((self.X[i,j] - cx)**2 + (self.Y[i,j] - cy)**2)
                    if dist_cyl < self.cylinder_radius * 3:
                        dist = min(dist, max(0.01, dist_cyl - self.cylinder_radius))
                    
                    # Strain rate
                    dudx = (self.u[i,j+1] - self.u[i,j-1]) / (2*self.dx)
                    dudy = (self.u[i+1,j] - self.u[i-1,j]) / (2*self.dy)
                    dvdx = (self.v[i,j+1] - self.v[i,j-1]) / (2*self.dx)
                    dvdy = (self.v[i+1,j] - self.v[i-1,j]) / (2*self.dy)
                    S = np.sqrt(2*(dudx**2 + dvdy**2) + (dudy + dvdx)**2)
                    
                    # Simple eddy viscosity
                    self.nu_tilde[i,j] = 0.09 * dist**2 * S
                    self.nu_tilde[i,j] = min(self.nu_tilde[i,j], 0.1)  # Limit
    
    def step(self):
        """Single time step using simple SIMPLE algorithm"""
        nu = self.mu / self.rho
        
        # Get eddy viscosity
        self.solve_spalart_allmaras_simple()
        nu_total = nu + self.nu_tilde
        
        # Store old values
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        # Momentum predictor (simple upwind)
        u_star = np.zeros_like(self.u)
        v_star = np.zeros_like(self.v)
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if not self.solid[i,j]:
                    # U-momentum
                    # Upwind convection
                    if self.u[i,j] > 0:
                        dudx = (self.u[i,j] - self.u[i,j-1]) / self.dx
                    else:
                        dudx = (self.u[i,j+1] - self.u[i,j]) / self.dx
                    
                    if self.v[i,j] > 0:
                        dudy = (self.u[i,j] - self.u[i-1,j]) / self.dy
                    else:
                        dudy = (self.u[i+1,j] - self.u[i,j]) / self.dy
                    
                    # Diffusion
                    d2udx2 = (self.u[i,j+1] - 2*self.u[i,j] + self.u[i,j-1]) / self.dx**2
                    d2udy2 = (self.u[i+1,j] - 2*self.u[i,j] + self.u[i-1,j]) / self.dy**2
                    
                    # Pressure gradient
                    dpdx = (self.p[i,j+1] - self.p[i,j-1]) / (2*self.dx)
                    
                    u_star[i,j] = self.u[i,j] + self.dt * (
                        -self.u[i,j]*dudx - self.v[i,j]*dudy +
                        nu_total[i,j]*(d2udx2 + d2udy2) - dpdx/self.rho
                    )
                    
                    # V-momentum (similar)
                    if self.u[i,j] > 0:
                        dvdx = (self.v[i,j] - self.v[i,j-1]) / self.dx
                    else:
                        dvdx = (self.v[i,j+1] - self.v[i,j]) / self.dx
                    
                    if self.v[i,j] > 0:
                        dvdy = (self.v[i,j] - self.v[i-1,j]) / self.dy
                    else:
                        dvdy = (self.v[i+1,j] - self.v[i,j]) / self.dy
                    
                    d2vdx2 = (self.v[i,j+1] - 2*self.v[i,j] + self.v[i,j-1]) / self.dx**2
                    d2vdy2 = (self.v[i+1,j] - 2*self.v[i,j] + self.v[i-1,j]) / self.dy**2
                    
                    dpdy = (self.p[i+1,j] - self.p[i-1,j]) / (2*self.dy)
                    
                    v_star[i,j] = self.v[i,j] + self.dt * (
                        -self.u[i,j]*dvdx - self.v[i,j]*dvdy +
                        nu_total[i,j]*(d2vdx2 + d2vdy2) - dpdy/self.rho
                    )
        
        # Under-relaxation
        self.u = (1-self.omega)*u_old + self.omega*u_star
        self.v = (1-self.omega)*v_old + self.omega*v_star
        
        # Pressure correction (simplified)
        div = np.zeros((self.ny, self.nx))
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if not self.solid[i,j]:
                    div[i,j] = (self.u[i,j+1] - self.u[i,j-1])/(2*self.dx) + \
                               (self.v[i+1,j] - self.v[i-1,j])/(2*self.dy)
        
        # Simple pressure update
        p_corr = np.zeros_like(self.p)
        for _ in range(10):  # Few iterations only
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    if not self.solid[i,j]:
                        p_corr[i,j] = 0.25 * (
                            p_corr[i+1,j] + p_corr[i-1,j] +
                            p_corr[i,j+1] + p_corr[i,j-1] -
                            self.dx*self.dy*self.rho*div[i,j]/self.dt
                        )
        
        # Update pressure
        self.p += self.omega * p_corr
        
        # Velocity correction
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if not self.solid[i,j]:
                    self.u[i,j] -= self.dt/(2*self.dx*self.rho) * (p_corr[i,j+1] - p_corr[i,j-1])
                    self.v[i,j] -= self.dt/(2*self.dy*self.rho) * (p_corr[i+1,j] - p_corr[i-1,j])
        
        # Apply BC
        self.apply_bc()
        
        # Compute residual
        max_div = np.max(np.abs(div[~self.solid]))
        return max_div
    
    def solve_step(self):
        """Wrapper for solver step"""
        if self.iteration < self.max_iter and self.solving:
            residual = self.step()
            self.residuals.append(residual)
            self.iteration += 1
            
            if residual < self.tolerance or self.iteration >= self.max_iter:
                self.solving = False
                return True
        return False

class SimpleCFDVisualization:
    def __init__(self):
        self.solver = SimpleCFDSolver()
        self.setup_figure()
        self.animation = None
        
    def setup_figure(self):
        """Setup figure"""
        self.fig, (self.ax_mesh, self.ax_residual) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initial mesh
        self.draw_mesh()
        
        # Residual plot
        self.ax_residual.set_title('Convergence')
        self.ax_residual.set_xlabel('Iteration')
        self.ax_residual.set_ylabel('Residual')
        self.ax_residual.set_yscale('log')
        self.ax_residual.grid(True, alpha=0.3)
        self.residual_line, = self.ax_residual.plot([], [], 'b-', linewidth=2)
        self.ax_residual.set_xlim(0, 500)
        self.ax_residual.set_ylim(1e-2, 1)
        
        # Controls
        plt.subplots_adjust(bottom=0.2)
        
        ax_velocity = plt.axes([0.2, 0.1, 0.3, 0.03])
        self.slider_velocity = Slider(ax_velocity, 'Velocity', 0.5, 2.0, 
                                      valinit=self.solver.inlet_velocity)
        self.slider_velocity.on_changed(self.update_velocity)
        
        ax_solve = plt.axes([0.6, 0.1, 0.1, 0.04])
        self.btn_solve = Button(ax_solve, 'Solve')
        self.btn_solve.on_clicked(self.start_solve)
        
        ax_reset = plt.axes([0.75, 0.1, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)
        
        self.status = self.fig.text(0.5, 0.02, 'Ready', ha='center')
    
    def draw_mesh(self):
        """Draw mesh"""
        self.ax_mesh.clear()
        
        # Remove any existing colorbars
        while self.ax_mesh.collections:
            self.ax_mesh.collections[0].remove()
        
        # Draw grid lines (sparse)
        for i in range(0, self.solver.ny, 3):
            self.ax_mesh.plot(self.solver.X[i,:], self.solver.Y[i,:], 
                            'k-', linewidth=0.2, alpha=0.3)
        for j in range(0, self.solver.nx, 3):
            self.ax_mesh.plot(self.solver.X[:,j], self.solver.Y[:,j], 
                            'k-', linewidth=0.2, alpha=0.3)
        
        # Cylinder
        circle = Circle(self.solver.cylinder_center, self.solver.cylinder_radius,
                       color='gray', zorder=10)
        self.ax_mesh.add_patch(circle)
        
        self.ax_mesh.set_xlim(0, self.solver.L)
        self.ax_mesh.set_ylim(0, self.solver.H)
        self.ax_mesh.set_aspect('equal')
        self.ax_mesh.set_title(f'Mesh ({self.solver.nx}×{self.solver.ny})')
        
        # Clear any colorbars from figure
        for ax in self.fig.axes:
            if ax != self.ax_mesh and ax != self.ax_residual:
                ax.remove()
    
    def update_velocity(self, val):
        self.solver.inlet_velocity = val
        self.solver.initialize_flow()
    
    def start_solve(self, event):
        if not self.solver.solving:
            self.solver.solving = True
            self.solver.iteration = 0
            self.solver.residuals = []
            self.status.set_text('Starting...')
            plt.draw()  # Force immediate update
            
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            
            # Fast animation
            self.animation = FuncAnimation(self.fig, self.update, interval=100,
                                         blit=False, cache_frame_data=False)
            plt.draw()  # Force start
    
    def reset(self, event):
        """Reset solver and clear everything properly"""
        # Stop animation
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        # Remove colorbar axes if they exist
        for ax in self.fig.axes[:]:
            if ax not in [self.ax_mesh, self.ax_residual]:
                try:
                    self.fig.delaxes(ax)
                except:
                    pass
        
        # Reset solver
        self.solver = SimpleCFDSolver()
        self.solver.inlet_velocity = self.slider_velocity.val
        self.solver.initialize_flow()
        
        # Redraw mesh
        self.draw_mesh()
        self.residual_line.set_data([], [])
        self.status.set_text('Reset')
        
        plt.draw()
    
    def update(self, frame):
        if self.solver.solving:
            # Run many iterations per frame
            for _ in range(20):
                done = self.solver.solve_step()
                if done:
                    break
            
            # Update every 20 iterations
            if self.solver.iteration % 20 == 0:
                n = len(self.solver.residuals)
                if n > 0:
                    self.residual_line.set_data(range(n), self.solver.residuals)
                    self.ax_residual.set_xlim(0, max(n+10, 100))
                    
                    self.status.set_text(f'Iter: {self.solver.iteration}, '
                                       f'Res: {self.solver.residuals[-1]:.3e}')
                    
                    if self.solver.iteration % 100 == 0:
                        print(f"Iteration {self.solver.iteration}: {self.solver.residuals[-1]:.3e}")
            
            if not self.solver.solving:
                self.show_results()
                self.status.set_text('Done!')
                if self.animation:
                    self.animation.event_source.stop()
    
    def show_results(self):
        """Show flow field with vorticity"""
        # Clear previous plot
        self.ax_mesh.clear()
        
        # Compute vorticity (rotation)
        vorticity = np.zeros_like(self.solver.u)
        for i in range(1, self.solver.ny-1):
            for j in range(1, self.solver.nx-1):
                if not self.solver.solid[i,j]:
                    dvdx = (self.solver.v[i,j+1] - self.solver.v[i,j-1]) / (2*self.solver.dx)
                    dudy = (self.solver.u[i+1,j] - self.solver.u[i-1,j]) / (2*self.solver.dy)
                    vorticity[i,j] = dvdx - dudy
        
        # Mask solid regions
        vorticity[self.solver.solid] = np.nan
        
        # Velocity magnitude
        vel = np.sqrt(self.solver.u**2 + self.solver.v**2)
        vel[self.solver.solid] = np.nan
        
        # Velocity contours
        levels = np.linspace(0, np.nanmax(vel), 20)
        cont = self.ax_mesh.contourf(self.solver.X, self.solver.Y, vel,
                                     levels=levels, cmap='coolwarm', alpha=0.8)
        
        # Add velocity colorbar in a specific location
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=self.fig, width_ratios=[3, 1, 0.3])
        cax1 = self.fig.add_subplot(gs[0, 2])  # Top right for velocity
        cbar1 = plt.colorbar(cont, cax=cax1)
        cbar1.set_label('Velocity [m/s]', fontsize=9)
        
        # Prepare for streamlines and quiver
        u_plot = self.solver.u.copy()
        v_plot = self.solver.v.copy()
        u_plot[self.solver.solid] = np.nan
        v_plot[self.solver.solid] = np.nan
        
        # Streamlines
        try:
            self.ax_mesh.streamplot(self.solver.X, self.solver.Y, u_plot, v_plot,
                                   density=1.0, color='white', linewidth=0.4)
        except:
            pass
        
        # Quiver plot colored by vorticity/turbulence
        step = 3
        X_q = self.solver.X[::step, ::step]
        Y_q = self.solver.Y[::step, ::step]
        U_q = u_plot[::step, ::step]
        V_q = v_plot[::step, ::step]
        vort_q = vorticity[::step, ::step]
        
        # Turbulence intensity
        turb_intensity = np.abs(vort_q) + self.solver.nu_tilde[::step, ::step] * 10
        
        # Create quiver plot with color based on turbulence
        quiv = self.ax_mesh.quiver(X_q, Y_q, U_q, V_q, turb_intensity,
                                   scale=25, width=0.003, cmap='hot',
                                   clim=[0, np.nanmax(np.abs(vorticity))*0.5])
        
        # Add vorticity colorbar below velocity colorbar
        cax2 = self.fig.add_subplot(gs[1, 2])  # Bottom right for vorticity
        cbar2 = plt.colorbar(quiv, cax=cax2)
        cbar2.set_label('Vorticity/Turb.', fontsize=9)
        
        # Cylinder
        circle = Circle(self.solver.cylinder_center, self.solver.cylinder_radius,
                       color='black', zorder=10)
        self.ax_mesh.add_patch(circle)
        
        # Add text annotation
        max_vort = np.nanmax(np.abs(vorticity))
        if not np.isnan(max_vort):
            self.ax_mesh.text(0.02, 0.98, f'Max Vorticity: {max_vort:.3f}',
                            transform=self.ax_mesh.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='white', alpha=0.7))
        
        # Maintain aspect ratio and limits
        self.ax_mesh.set_xlim(0, self.solver.L)
        self.ax_mesh.set_ylim(0, self.solver.H)
        self.ax_mesh.set_aspect('equal', adjustable='box')
        self.ax_mesh.set_title('Flow Field with Spalart-Allmaras Turbulence\n(Arrows colored by vorticity/turbulence)', fontsize=12)
        self.ax_mesh.set_xlabel('X [m]', fontsize=10)
        self.ax_mesh.set_ylabel('Y [m]', fontsize=10)
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    print("LIGHTWEIGHT CFD SOLVER WITH SPALART-ALLMARAS")
    print("-" * 40)
    print("Features:")
    print("• 60×30 stretched grid")
    print("• Simplified SA turbulence model")  
    print("• Fast SIMPLE algorithm")
    print("• Web-friendly performance")
    print("-" * 40)
    
    viz = SimpleCFDVisualization()
    viz.show()