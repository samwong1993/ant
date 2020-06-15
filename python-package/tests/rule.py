#test whether the sample satsifies the given rule
import math
def rule_decide(rule,data):
    rule = rule[10:]
    rule = rule.split('and')
    b = rule.pop()
    rule.append(b.split('then')[0])
    results = b.split('then')[1]
    for i in range(len(rule)):
        if ')' in rule[i]:
            a = rule[i].replace('(', '')
            a = a.replace(')', '')
            a = a.split('or')
            rule[i] = a
    out = True
    for i in range(len(rule)):
        if isinstance(rule[i], list):
            log = False
            for j in range(2):
                log = log | rule_single(rule[i][j],data)
        else:
            log = rule_single(rule[i], data)
        out = out & log
    return out,results


def rule_single(rule,data):
    if rule[0] == ' ':
        rule = rule[1:]
    temp = rule.split(' ')
    left = temp[0]
    cond = temp[1]
    righ = temp[2]
    left = data[left]
    if cond == 'is':
        return(math.isnan(left))
    else:
        return(eval(str(left) + cond + righ))
