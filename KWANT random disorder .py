#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kwant
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm 
import matplotlib.colors as mcolors 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

lat = kwant.lattice.honeycomb()
a, b = lat.sublattices  

def make_graphene_system(r=100, t=-2.7, n=0.0, seed=12):
    random.seed(seed) 
    syst = kwant.Builder()

    def square(pos):
        x, y = pos
        return (-r < x < r) and (-r < y < r)

    syst[lat.shape(square, (0, 0))] = 0.0
    syst[lat.neighbors()] = t

    if n > 0:
        for site in list(syst.sites()):
            if random.random() < n:
                del syst[site]
    syst.eradicate_dangling()
    return syst.finalized()

def compute_dos(systems, energy_range=(-4, 4), num_energy_points=300):
    dos_data = []
    energy_grid = np.linspace(energy_range[0], energy_range[1], num_energy_points)
    for syst in systems:
        spectrum = kwant.kpm.SpectralDensity(syst, num_moments=100, rng=0)
        energies, densities = spectrum()
        interpolated_dos = np.interp(energy_grid, energies, densities.real)
        dos_data.append(interpolated_dos)
    return energy_grid, np.array(dos_data)

def plot_dos_contour(energy_grid, dos_data, vacancy_levels):
    X, Y = np.meshgrid(energy_grid, vacancy_levels)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.figure(figsize=(7, 5))
    contour = plt.contourf(X, Y, dos_data, levels=100, cmap="viridis")
    plt.colorbar(contour, label="DOS")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Vacancy Concentration")
    plt.title("DOS vs Vacancy Concentration")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig("/your path/contour.png", dpi=300)
    plt.show()

def plot_wavefunction(syst, state_index=0, filename=None):
    # Sparse Hamiltonian
    ham = csr_matrix(syst.hamiltonian_submatrix(sparse=True))

    # Compute only the lowest-energy state
    evals, evecs = eigsh(ham, k=1, which="SA")
    E = evals[state_index]
    wf = np.abs(evecs[:, state_index]) ** 2
    wf /= np.sum(wf)  # normalize

    positions = np.array([site.pos for site in syst.sites])
    x, y = positions[:, 0], positions[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        x, y, c=wf, cmap="inferno", s=40, edgecolor="none",
        norm=mcolors.LogNorm(vmin=wf.max() * 1e-4, vmax=wf.max())
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("|ψ|² (normalized)")
    ax.set_title(f"Ground-State Wavefunction (E = {E:.3f} eV)", fontsize=12)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.axis("equal")
    plt.tight_layout()
    if filename:
        plt.savefig(f"/your path/{filename}.png", dpi=300)
    plt.show()

def plot_system(syst, title, filename=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    kwant.plot(syst, ax=ax)
    ax.set_title(title)
    if filename:
        plt.savefig(f"/your path/{filename}.png", dpi=300)
    plt.show()

def plot_all_dos(systems, labels, file_path="/your path/dos_data.xlsx"):
    plt.figure(figsize=(6, 6))
    dos_data = {}
    for syst, label in zip(systems, labels):
        spectrum = kwant.kpm.SpectralDensity(syst, num_moments=100, rng=0)
        energies, densities = spectrum()
        dos_data["Energy (eV)"] = energies
        dos_data[label] = densities.real
        plt.plot(energies, densities.real, linewidth=1, label=label)
    df = pd.DataFrame(dos_data)
    df.to_excel(file_path, index=False)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.legend()
    plt.tick_params(axis="both", direction="in", length=5, width=1.2, top=True, right=True)
    plt.grid(False)
    plt.xlim(-8, 8)
    plt.tight_layout()
    plt.savefig(f"/your path/dos.png", dpi=300)
    plt.show()

def main():
    vacancy_levels = [0, 0.003,0.00835,0.03,0.045]#change accordingly
    systems = {}
    for v in vacancy_levels:
        syst = make_graphene_system(r=100, n=v)
        systems[v] = syst
        num_atoms = len(syst.sites)
        print(f"Vacancy {v:.3f} → Total atoms: {num_atoms}")
        plot_system(syst, f"Graphene with {v:.1%} vacancies", filename=f"graphene_defect_{v}")
    energy_grid, dos_data = compute_dos(list(systems.values()), energy_range=(-1, 1), num_energy_points=100)
    plot_dos_contour(energy_grid, dos_data, vacancy_levels)
    plot_all_dos(list(systems.values()), labels=[f"{v*100:.1f}% Vacancies" for v in vacancy_levels])
    for v in vacancy_levels:
        plot_wavefunction(systems[v], filename=f"wavefunction_{v}")

if __name__ == "__main__":
    main()

