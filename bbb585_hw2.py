#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:52:02 2019

@author: Zach Sheldon
"""

import numpy as np
import matplotlib.pyplot as plt

## GOAL - to numerically simulate a Hodgkin-Huxley single-compartment neuron with dynamical voltage-dependent functions
## first I will explore the nonlinear voltage-gated functions
## then I will fully simulate the model
## all of the equations and parameters are from Abbott & Dayan's Theoretical Neuroscience textbook

## activation probabilities n(t) and m(t) and de-inactivation probability h(t):
## dn/dt = alpha_n(V)*(1 - n) - beta_n(V)*n
## dm/dt = alpha_m(V)*(1 - n) - beta_m(V)*n
## dh/dt = alpha_h(V)*(1 - n) - beta_h(V)*n

# opening/close rate functions
def alpha_n(voltage):
    voltage = voltage * 1000 # convert to mV
    numerator = 0.01 * (voltage + 55)
    denominator = 1.0 - np.exp(-0.1 * (voltage + 55))
    return (numerator / denominator) * 1000 # return 1 / sec
def beta_n(voltage):
    voltage = voltage * 1000 # convert to mV
    return (0.125 * np.exp(-0.0125 * (voltage + 65))) * 1000 # return 1 / sec
def alpha_m(voltage):
    voltage = voltage * 1000 # convert to mV    
    numerator = 0.1 * (voltage + 40)
    denominator = 1 - np.exp(-0.1 * (voltage + 40))
    return (numerator / denominator) * 1000 # return 1 / sec
def beta_m(voltage):
    voltage = voltage * 1000 # convert to mV
    return (4.0 * np.exp(-0.0556 * (voltage + 65))) * 1000 # return 1 / sec
def alpha_h(voltage):
    voltage = voltage * 1000 # convert to mV   
    return (0.07 * np.exp(-0.05 * (voltage + 65))) * 1000 # return 1 / sec
def beta_h(voltage):
    voltage = voltage * 1000 # convert to mV
    return (1 / (1 + np.exp(-0.1 * (voltage + 35)))) * 1000 # return 1 / sec

# time constant functions
def tau_n(voltage):
    return 1 / (alpha_n(voltage) + beta_n(voltage))
def tau_m(voltage):
    return 1 / (alpha_m(voltage) + beta_m(voltage))
def tau_h(voltage):
    return 1 / (alpha_h(voltage) + beta_h(voltage))

# steady-state activation functions (n_infinity, m_infinity, h_infinity)
def n_inf(voltage):
    return (alpha_n(voltage) / (alpha_n(voltage) + beta_n(voltage)))
def m_inf(voltage):
    return (alpha_m(voltage) / (alpha_m(voltage) + beta_m(voltage)))
def h_inf(voltage):
    return (alpha_h(voltage) / (alpha_h(voltage) + beta_h(voltage)))

## plot voltage-dependent gating functions (replicating figures 5.9 and 5.10 from Dayan & Abbott's Theoretical Neuroscience textbook)
voltage_arr = np.arange(-0.100, 0, 0.001) # voltage range - V
alpha_n_arr = alpha_n(voltage_arr)
beta_n_arr = beta_n(voltage_arr)
n_inf_arr = n_inf(voltage_arr)
m_inf_arr = m_inf(voltage_arr)
h_inf_arr = h_inf(voltage_arr)
tau_n_arr = tau_n(voltage_arr)
tau_m_arr = tau_m(voltage_arr)
tau_h_arr = tau_h(voltage_arr)

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(voltage_arr*1000, alpha_n_arr/1000, label='alpha_n')
plt.plot(voltage_arr*1000, beta_n_arr/1000, label='beta_n')
plt.legend()
plt.xlabel('V (mV)')
plt.ylabel('Alpha or Beta (1 / msec)')
plt.subplot(132)
plt.plot(voltage_arr*1000, n_inf_arr, label='n_infinity', color='r')
plt.ylim([0, 1])
plt.xlabel('V (mv)')
plt.ylabel('N_infinity')
plt.legend()
plt.title('Voltage-Dependent Gating Functions - Potassium')
plt.subplot(133)
plt.plot(voltage_arr*1000, tau_n_arr*1000, label='tau_n', color='g')
plt.xlabel('V (mv)')
plt.ylabel('tau_n')
plt.legend()
plt.tight_layout()
#plt.savefig('voltage_dependent_gating_funcs_potassium.png')

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(voltage_arr*1000, n_inf_arr, label='n_infinity')
plt.plot(voltage_arr*1000, m_inf_arr, label='m_infinity')
plt.plot(voltage_arr*1000, h_inf_arr, label='h_infinity')
plt.ylim([0, 1])
plt.xlabel('V (mv)')
plt.legend()
plt.title('Voltage-Dependent Gating Functions')
plt.subplot(122)
plt.plot(voltage_arr*1000, tau_n_arr*1000, label='tau_n')
plt.plot(voltage_arr*1000, tau_m_arr*1000, label='tau_m')
plt.plot(voltage_arr*1000, tau_h_arr*1000, label='tau_h')
plt.xlabel('V (mv)')
plt.ylabel('tau (mS)')
plt.title('Time Constant - Tau')
plt.legend()
plt.tight_layout()
#plt.savefig('voltage_dependent_gating_funcs_tau.png')

# full Hodgkin-Huxley model: c_m * (dV/dt) = -i_m + (I_e/A)
# which is equivalent to saying: dV = (-i_m + (I_e/A)) * (dt/cm)
# i_m = g_leak(V - E_leak) + g_k*n**4(V - E_k) + g_na*(m**3)*h(V - E_na)

# parameters
g_leak = 0.003 * (10**-3) * (10**6) # S/m**2
g_k = 0.36 * (10**-3) * (10**6) # S/m**2
g_na = 1.2 * (10**-3) * (10**6)  # S/m**2
E_leak = -54.387 * (10**-3) # V
E_k = -77.0 * (10**-3) # V
E_na = 50.0 * (10**-3)  # V
c_m = 10.0 * (10**-9) * (10**6) # F/m**2
A = 0.1 * (10**-6) # m**2
v_0 = -65.0 * (10**-3) # resting potential V(t=0) - V
n_0 = n_inf(v_0) # initial value for n 
m_0 = m_inf(v_0) # initial value for m
h_0 = h_inf(v_0) # initial value for h
dt = 0.00001 # sec 
t_arr = np.arange(0,0.015,dt) # array with time steps in msec
num_t_steps = len(t_arr)
I_e = np.zeros(num_t_steps) # microAmps
# input current starting at t = 0.005 sec
for i in range(0, len(I_e)):
    curr_time = t_arr[i]
    if curr_time >= 0.005:
        I_e[i] = 20 * (10**-9) # alter this value to change the input current - 20 nA

# value updating function
def hodgkin_huxley_simulation(I_e, dt, num_t_steps, g_leak, g_k, g_na, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0, h_0):
    # lists containing all the values we want to keep track of and plot later
    volt_arr = []
    n_arr = []
    m_arr = []
    h_arr = []
    i_m_arr = []
    # loop through time steps and calculate new values
    for i in range(0, num_t_steps):
        if i == 0:
            # initial values
            volt_arr.append(v_0)
            n_arr.append(n_0)
            m_arr.append(m_0)
            h_arr.append(h_0)
        else:
            # previous values
            prev_volt = volt_arr[i-1]
            prev_n = n_arr[i-1]
            prev_m = m_arr[i-1]
            prev_h = h_arr[i-1]
            # change in gating variables
            dn = ((alpha_n(prev_volt)*(1 - prev_n)) - (beta_n(prev_volt)*prev_n)) * dt
            dm = ((alpha_m(prev_volt)*(1 - prev_m)) - (beta_m(prev_volt)*prev_m)) * dt
            dh = ((alpha_h(prev_volt)*(1 - prev_h)) - (beta_h(prev_volt)*prev_h)) * dt
            # updated gating values
            curr_n = prev_n + dn
            curr_m = prev_m + dm
            curr_h = prev_h + dh
            # new membrane current
            curr_i_m = g_leak*(prev_volt - E_leak) + g_k*(curr_n**4)*(prev_volt - E_k) + g_na*(curr_m**3)*curr_h*(prev_volt - E_na)
            # change in voltage
            dV = (-curr_i_m + (I_e[i]/A)) * (dt/c_m)
            # new voltage
            curr_volt = prev_volt + dV
            # update lists
            volt_arr.append(curr_volt)
            n_arr.append(curr_n)
            m_arr.append(curr_m)
            h_arr.append(curr_h)
            i_m_arr.append(curr_i_m)
            
    return volt_arr, i_m_arr, n_arr, m_arr, h_arr

volt_arr, i_m_arr, n_arr, m_arr, h_arr = hodgkin_huxley_simulation(I_e, dt, num_t_steps, g_leak, g_k, g_na, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0, h_0)

# plot results to replicate figure 5.11 from textbook
plt.figure(figsize=(7,10))
plt.subplot(511)
plt.plot(t_arr, volt_arr, label='voltage', color='r')
plt.ylabel('Voltage')
plt.title('Hodgkin-Huxley Model')
plt.xlim([0, 0.015])
plt.legend()
plt.subplot(512)
plt.plot(t_arr[:num_t_steps-1], i_m_arr, label='membrane current', color='g')
plt.ylabel('i_m (Amps / m**2)')
plt.legend()
plt.xlim([0, 0.015])
plt.subplot(513)
plt.plot(t_arr, m_arr, label='m', color='y')
plt.legend()
plt.ylabel('m')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(514)
plt.plot(t_arr, h_arr, label='h', color='k')
plt.legend()
plt.ylabel('h')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(515)
plt.plot(t_arr, n_arr, label='n', color='b')
plt.ylabel('n')
plt.xlabel('Time (msec)')
plt.legend()
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.tight_layout()
#plt.savefig('hh_model_simulation.png')

# simulate hodgkin and huxley model with blocked channels (Na or K) to simulate the effects of a channel-blocking neurotoxin
volt_arr_block_na, i_m_arr_block_na, n_arr_block_na, m_arr_block_na, h_arr_block_na = hodgkin_huxley_simulation(I_e, dt, num_t_steps, g_leak, g_k, g_na/10, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0, h_0)
volt_arr_block_k, i_m_arr_block_k, n_arr_block_k, m_arr_block_k, h_arr_block_k = hodgkin_huxley_simulation(I_e, dt, num_t_steps, g_leak, g_k/10, g_na, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0, h_0)

# plot results
plt.figure(figsize=(7,10))
plt.subplot(511)
plt.plot(t_arr, volt_arr_block_na, label='voltage', color='r')
plt.ylabel('Voltage')
plt.title('Hodgkin-Huxley Model with Blocked Sodium Channels')
plt.xlim([0, 0.015])
plt.legend()
plt.subplot(512)
plt.plot(t_arr[:num_t_steps-1], i_m_arr_block_na, label='membrane current', color='g')
plt.ylabel('i_m (Amps / m**2)')
plt.legend()
plt.xlim([0, 0.015])
plt.subplot(513)
plt.plot(t_arr, m_arr_block_na, label='m', color='y')
plt.legend()
plt.ylabel('m')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(514)
plt.plot(t_arr, h_arr_block_na, label='h', color='k')
plt.legend()
plt.ylabel('h')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(515)
plt.plot(t_arr, n_arr_block_na, label='n', color='b')
plt.ylabel('n')
plt.xlabel('Time (msec)')
plt.legend()
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.tight_layout()
#plt.savefig('hh_model_simulation_block_na.png')

plt.figure(figsize=(7,10))
plt.subplot(511)
plt.plot(t_arr, volt_arr_block_k, label='voltage', color='r')
plt.ylabel('Voltage')
plt.title('Hodgkin-Huxley Model with Blocked Potassium Channels')
plt.xlim([0, 0.015])
plt.legend()
plt.subplot(512)
plt.plot(t_arr[:num_t_steps-1], i_m_arr_block_k, label='membrane current', color='g')
plt.ylabel('i_m (Amps / m**2)')
plt.legend()
plt.xlim([0, 0.015])
plt.subplot(513)
plt.plot(t_arr, m_arr_block_k, label='m', color='y')
plt.legend()
plt.ylabel('m')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(514)
plt.plot(t_arr, h_arr_block_k, label='h', color='k')
plt.legend()
plt.ylabel('h')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(515)
plt.plot(t_arr, n_arr_block_k, label='n', color='b')
plt.ylabel('n')
plt.xlabel('Time (msec)')
plt.legend()
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.tight_layout()
#plt.savefig('hh_model_simulation_block_k.png')

# simulate hodgkin and huxley model with persistent Sodium channels (no inactivation)
def hodgkin_huxley_simulation_persistentNa(I_e, dt, num_t_steps, g_leak, g_k, g_na, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0):
    # lists containing all the values we want to keep track of and plot later
    volt_arr = []
    n_arr = []
    m_arr = []
    i_m_arr = []
    # loop through time steps and calculate new values
    for i in range(0, num_t_steps):
        if i == 0:
            # initial values
            volt_arr.append(v_0)
            n_arr.append(n_0)
            m_arr.append(m_0)
        else:
            # previous values
            prev_volt = volt_arr[i-1]
            prev_n = n_arr[i-1]
            prev_m = m_arr[i-1]
            # change in gating variables
            dn = ((alpha_n(prev_volt)*(1 - prev_n)) - (beta_n(prev_volt)*prev_n)) * dt
            dm = ((alpha_m(prev_volt)*(1 - prev_m)) - (beta_m(prev_volt)*prev_m)) * dt
            # updated gating values
            curr_n = prev_n + dn
            curr_m = prev_m + dm
            # new membrane current
            curr_i_m = g_leak*(prev_volt - E_leak) + g_k*(curr_n**4)*(prev_volt - E_k) + g_na*(curr_m**4)*(prev_volt - E_na)
            # change in voltage
            dV = (-curr_i_m + (I_e[i]/A)) * (dt/c_m)
            # new voltage
            curr_volt = prev_volt + dV
            # update lists
            volt_arr.append(curr_volt)
            n_arr.append(curr_n)
            m_arr.append(curr_m)
            i_m_arr.append(curr_i_m)
            
    return volt_arr, i_m_arr, n_arr, m_arr

volt_arr_persist_na, i_m_arr_persist_na, n_arr_persist_na, m_arr_persist_na = hodgkin_huxley_simulation_persistentNa(I_e, dt, num_t_steps, g_leak, g_k, g_na, E_leak, E_k, E_na, c_m, A, v_0, n_0, m_0)

# plot results
plt.figure(figsize=(7,10))
plt.subplot(411)
plt.plot(t_arr, volt_arr_persist_na, label='voltage', color='r')
plt.ylabel('Voltage')
plt.title('Hodgkin-Huxley Model with Persistent Sodium Channels')
plt.xlim([0, 0.015])
plt.legend()
plt.subplot(412)
plt.plot(t_arr[:num_t_steps-1], i_m_arr_persist_na, label='membrane current', color='g')
plt.ylabel('i_m (Amps / m**2)')
plt.legend()
plt.xlim([0, 0.015])
plt.subplot(413)
plt.plot(t_arr, m_arr_persist_na, label='m', color='y')
plt.legend()
plt.ylabel('m')
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.subplot(414)
plt.plot(t_arr, n_arr_persist_na, label='n', color='b')
plt.ylabel('n')
plt.xlabel('Time (msec)')
plt.legend()
plt.ylim([0, 1])
plt.xlim([0, 0.015])
plt.tight_layout()
#plt.savefig('hh_model_simulation_persistent_na.png')