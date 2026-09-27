import json,sys, statistics

with open('./results/baseline.json') as f:
    baseline = json.load(f) 
print(baseline.keys())
print(baseline['by_agent'])
print(list(baseline))

## Number of rounds
print(len(baseline['by_round']))

# Steps mean and sd
rounds = list(baseline['by_round'].values())
steps = [r['steps'] for r in rounds]
print('Steps:', steps)
print('Steps mean:', statistics.mean(steps))
print('Steps sd:', statistics.stdev(steps))

# coins means and sd
coins = [c['coins'] for c in rounds]
print('coins:', coins)
print('coins mean:', statistics.mean(coins))
print('coins sd:', statistics.stdev(coins))


# suicide rate
suicides = [s['suicides'] for s in rounds if s['suicides']>0]
suicide_rate = len(suicides)/len(baseline['by_round'])*100
print(f"Suicide Rate:", suicide_rate,"%")

# invalid rate
invalid_number = baseline['by_agent']['smurfer']['invalid']
steps_number = baseline['by_agent']['smurfer']['steps']
print(f"Invalid Rate:",(invalid_number/steps_number)*100,"%")
