from src.topology import Topology

n_max = 30
l_max = 300
L = 0.2

print(n_max, l_max, L)
a = Topology(debug=False, l_max = l_max, n_max=n_max, L = L * 14 * 1e3, use_numba=True, fig_name = 'l_max_{}_n_max_{}'.format(l_max, n_max))
_, c_l_a = a.plot_alm_realizations()
a.calculate_C(only_diag=True)
'''
n_max=25
L=0.4
b = Topology(debug=False, l_max = l_max, n_max=n_max, L = L * 14 * 1e3, use_numba=True, fig_name = 'l_max_{}_n_max_{}'.format(l_max, n_max))
b.calculate_C(only_diag=True)

n_max=30
L=0.6
c = Topology(debug=False, l_max = l_max, n_max=n_max, L = L * 14 * 1e3, use_numba=True, fig_name = 'l_max_{}_n_max_{}'.format(l_max, n_max))
c.calculate_C(only_diag=True)'''

plt.figure()
plt.plot(get_D_l(a.C_TT_diag)[5:l_max], linewidth=4, label='L=0.2 True C_l')
plt.plot(get_D_l(a.powers[:, 0])[5:l_max], linewidth=4, label='CAMB')
for i in range(4):
    plt.plot(get_D_l(c_l_a[:, id])[5:l_max], label='Realization {}'.format(i))
#plt.plot(get_D_l(b.C_TT_diag)[5:l_max], label='L=0.4')
#plt.plot(get_D_l(c.C_TT_diag)[5:l_max], label='L=0.6')
plt.legend()
plt.ylabel(r'$\ell (\ell+1)C^{TT}_\ell / 2\pi \, [\mu K^2]$')
plt.xlabel(r'$\ell$')
plt.xscale('log')
plt.savefig('tmp.pdf')
#a.plot_C()
#a.make_realization()