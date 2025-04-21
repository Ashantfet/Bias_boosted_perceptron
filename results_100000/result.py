import os
import re

# Function to parse ChampSim output files
def parse_champsim_output(file_path):
    metrics = {}
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Extract IPC
        ipc_match = re.search(r'cumulative IPC: ([\d.]+)', content)
        metrics['IPC'] = float(ipc_match.group(1)) if ipc_match else None
        
        # Extract Branch Prediction Accuracy
        branch_acc_match = re.search(r'Branch Prediction Accuracy: ([\d.]+)%', content)
        metrics['Branch Accuracy (%)'] = float(branch_acc_match.group(1)) if branch_acc_match else None
        
        # Extract L1I Miss Rate
        l1i_miss_match = re.search(r'cpu0->cpu0_L1I TOTAL\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)', content)
        l1i_access_match = re.search(r'cpu0->cpu0_L1I TOTAL\s+ACCESS:\s+(\d+)', content)
        if l1i_miss_match and l1i_access_match:
            l1i_misses = int(l1i_miss_match.group(1))
            l1i_accesses = int(l1i_access_match.group(1))
            metrics['L1I Miss Rate (%)'] = (l1i_misses / l1i_accesses) * 100 if l1i_accesses > 0 else 0
        else:
            metrics['L1I Miss Rate (%)'] = None
        
        # Extract L1D Miss Rate
        l1d_miss_match = re.search(r'cpu0->cpu0_L1D TOTAL\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)', content)
        l1d_access_match = re.search(r'cpu0->cpu0_L1D TOTAL\s+ACCESS:\s+(\d+)', content)
        if l1d_miss_match and l1d_access_match:
            l1d_misses = int(l1d_miss_match.group(1))
            l1d_accesses = int(l1d_access_match.group(1))
            metrics['L1D Miss Rate (%)'] = (l1d_misses / l1d_accesses) * 100 if l1d_accesses > 0 else 0
        else:
            metrics['L1D Miss Rate (%)'] = None
        
        # Extract LLC Miss Rate
        llc_miss_match = re.search(r'cpu0->LLC TOTAL\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)', content)
        llc_access_match = re.search(r'cpu0->LLC TOTAL\s+ACCESS:\s+(\d+)', content)
        if llc_miss_match and llc_access_match:
            llc_misses = int(llc_miss_match.group(1))
            llc_accesses = int(llc_access_match.group(1))
            metrics['LLC Miss Rate (%)'] = (llc_misses / llc_accesses) * 100 if llc_accesses > 0 else 0
        else:
            metrics['LLC Miss Rate (%)'] = None
        
        # Extract DRAM Row Buffer Misses
        dram_rb_miss_match = re.search(r'ROW_BUFFER_MISS:\s+(\d+)', content)
        metrics['DRAM Row Buffer Misses'] = int(dram_rb_miss_match.group(1)) if dram_rb_miss_match else None
    
    return metrics

# Directory containing the files
directory = r'C:\Users\86779\Desktop\ChampSim_Results\results'

# Filter files based on naming patterns
ashant_files = [f for f in os.listdir(directory) if f.endswith('_ashant.txt')]
hybrid_files = [f for f in os.listdir(directory) if f.endswith('_sandeep.txt')]
perceptron_files = [f for f in os.listdir(directory) if f.endswith('_perceptron.txt')]
a_files = [f for f in os.listdir(directory) if f.startswith('a_')]

# Parse all files
ashant_results = {file: parse_champsim_output(os.path.join(directory, file)) for file in ashant_files}
hybrid_results = {file: parse_champsim_output(os.path.join(directory, file)) for file in hybrid_files}
perceptron_results = {file: parse_champsim_output(os.path.join(directory, file)) for file in perceptron_files}
a_results = {file: parse_champsim_output(os.path.join(directory, file)) for file in a_files}

# Print results
print("Ashant Results:")
for file, metrics in ashant_results.items():
    print(f"{file}: {metrics}")

print("\Sandeep Results:")
for file, metrics in hybrid_results.items():
    print(f"{file}: {metrics}")

print("\nPerceptron Results:")
for file, metrics in perceptron_results.items():
    print(f"{file}: {metrics}")

print("\na_*.txt Results:")
for file, metrics in a_results.items():
    print(f"{file}: {metrics}")