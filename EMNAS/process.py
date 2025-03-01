import copy
AR = [['11'],
      ['110','101','011'],
      ['1100','1010','0110','1001','0101','0011'],
      ['11000','10100','01100','10010','01010','00110','10001','01001','00101','00011'],
      ['110000','101000','011000','100100','010100','001100','100010','010010','001010','000110',
       '100001','010001','001001','000101','000011'],
      ['1100000', '1010000', '0110000', '1001000', '0101000', '0011000', '1000100', '0100100', '0010100', '0001100',
       '1000010', '0100010', '0010010', '0001010', '0000110','1000001','0100001','0010001','0001001','0000101','0000011']]

Ar_All = []
for j,ar1 in enumerate(AR[1]):
    for k,ar2 in enumerate(AR[2]):
        for z,ar3 in enumerate(AR[3]):
            for q, ar4 in enumerate(AR[4]):
                for p, ar5 in enumerate(AR[5]):
                    ar = ['11',ar1,ar2,ar3,ar4,ar5]
                    Ar_All.append('-'.join(ar))


def s(NS):
    l = len(NS)
    if l == 0:
        return []
    a = copy.deepcopy(NS)[::-1]
    t = []
    ts = []
    re = {}
    search = copy.deepcopy(NS)
    for n,(ar,node) in enumerate(a):
        if A[node] == NS[:l-n]:
            ts.append((ar,node))
            search.remove((ar,node))
        else:
            t.append(A[node])
            for j in A[node]:
                if j in search:
                    search.remove(j)
            break

    while search != []:
        n = search[-1][1]
        t.append(A[n])
        for j in A[n]:
            if j in search:
                search.remove(j)

    if t != []:
        re['split'] = t
    if ts != []:
        ts = sorted(ts, key=lambda stu: stu[1])
        re['guding'] = ts

    return re


def search(NS):
    if len(NS) == 0:
        return []
    r = s(NS)
    t = []
    try:
        if 'split' in r:
            for n,i in enumerate(r['split']):
                t.append(search(i))
            r['split'] = t
        # if a[0] != []:
        #     for n,i in enumerate(a[0]):
        #         t.append(search(i))
            return r
        else:
            return r
    except:
        print(r)




SS = {}
def hebing(sp):
    m = []
    if 'split' in sp:
        for i in sp['split']:
            a = hebing(i)
            l = len(a)
            k = 0
            for j in a:
                k = k + int(j[0],2)
            m.append([a,l,k])
        m = sorted(m, key=lambda stu: (stu[1], stu[2]),reverse=True)

        S = {}
        if len(m) > 1:
            for i in m:
                if len(i[0])>1:
                    b1 = []
                    for ar1 in i[0]:
                        b1.append(ar1[0])
                    b1 = ['-'.join(b1),i[0][-1][1]]
                else:
                    b1 = [i[0][0][0],i[0][0][1]]
                for j in m:
                    if (i != j) and (len(i) == len(j)):
                        if len(j[0]) > 1:
                            b2 = []
                            for ar1 in j[0]:
                                b2.append(ar1[0])
                            b2 = ['-'.join(b2),i[0][-1][1]]
                        else:
                            b2 = [j[0][0][0],j[0][0][1]]

                        if b1[0] == b2[0]:
                            if b1[0] not in S:
                                S[b1[0]] = [b1[1],b2[1]]
                            else:
                                S[b1[0]].append(b2[1])
            for i in S:
                q = sorted(S[i])
                for j in q:
                    SS[j] = q

    if 'guding' in sp:
        m.append([sp['guding'],0,0])

    T = []
    M = []
    if len(m) > 1:
        for n,i in enumerate(m):
            if len(i[0]) > 1:
                for j in i[0]:
                    if j[1] not in T:
                        op = ['0'] * (len(T)+2)
                        T.append(j[1])

                        t = 0
                        for node,o in enumerate(j[0]):
                            if o == '1':
                                t = t + 1
                                if node-2 >= 0:
                                    K = 0
                                    for new_side,k in enumerate(T):
                                        if len(j) == 3:
                                            pnode = i[0][node - 2][1]
                                            if pnode in SS:
                                                if (k in SS[pnode]) and (K == 0):
                                                    # if k == i[0][node - 2][1]:
                                                    if op[new_side + 2] == '0':
                                                        op[new_side + 2] = '1'
                                                        K = 1
                                            else:
                                                if k == pnode:
                                                # if k == i[0][node - 2][1]:
                                                    op[new_side+2] = '1'
                                        else:
                                            pnode = node - 2
                                            if pnode in SS:
                                                if (k in SS[pnode]) and (K == 0):
                                                    # if k == i[0][node - 2][1]:
                                                    if op[new_side + 2] == '0':
                                                        op[new_side + 2] = '1'
                                                        K = 1
                                            else:
                                                if k == pnode:
                                                # if k == i[0][node - 2][1]:
                                                    op[new_side+2] = '1'

                                            # if k == (node - 2):
                                            #     op[new_side+2] = '1'
                                else:
                                    op[node] = '1'

                            if t == 2:
                                M.append((''.join(op),j[1],1))
                                break
            else:
                if i[0][0][1] not in T:
                    op = ['0'] * (len(T) + 2)
                    T.append(i[0][0][1])

                    t = 0
                    for node, o in enumerate(i[0][0][0]):
                        if o == '1':
                            t = t + 1
                            if node - 2 >= 0:
                                for new_side, k in enumerate(T):
                                    if k == (node - 2):
                                        op[new_side + 2] = '1'
                            else:
                                op[node] = '1'

                        if t == 2:
                            M.append((''.join(op), i[0][0][1],1))
                            break
    else:
        if len(m[0][0])>1:
            for j in m[0][0]:
                # print(j)
                if j[1] not in T:
                    op = ['0'] * (len(T) + 2)
                    T.append(j[1])

                    t = 0
                    for node, o in enumerate(j[0]):
                        if o == '1':
                            t = t + 1
                            if node - 2 >= 0:

                                # if pnode in SS:
                                #     pnode = SS[pnode]
                                K = 0
                                for new_side, k in enumerate(T):
                                    if len(j) == 3:
                                        pnode = m[0][0][node - 2][1]
                                        if pnode in SS:
                                            if (k in SS[pnode]) and (K == 0):
                                                # if k == i[0][node - 2][1]:
                                                if op[new_side + 2] == '0':
                                                    op[new_side + 2] = '1'
                                                    K = 1
                                        else:
                                            pnode = node - 2
                                            if pnode in SS:
                                                if (k in SS[pnode]) and (K == 0):
                                                    # if k == i[0][node - 2][1]:
                                                    if op[new_side + 2] == '0':
                                                        op[new_side + 2] = '1'
                                                        K = 1
                                            else:
                                                if k == pnode:
                                                    # if k == i[0][node - 2][1]:
                                                    op[new_side + 2] = '1'

                                            # if k == pnode:
                                            #     # if k == i[0][node - 2][1]:
                                            #     op[new_side + 2] = '1'

                                        # if ((k == pnode) or (k in SS[pnode])):
                                        # # if k == m[0][0][node - 2][1]:
                                        #     op[new_side+2] = '1'
                                    else:
                                        if k == (node - 2):
                                            op[new_side+2] = '1'

                            else:
                                op[node] = '1'

                        if t == 2:
                            M.append((''.join(op), j[1],1))
                            break
        else:
            if m[0][0][0][1] not in T:
                op = ['0'] * (len(T) + 2)
                T.append(m[0][0][0][1])

                t = 0
                for node, o in enumerate(m[0][0][0][0]):
                    if o == '1':
                        t = t + 1
                        if node - 2 >= 0:
                            for new_side, k in enumerate(T):
                                if k == (node - 2):
                                    op[new_side+2] = '1'
                        else:
                            op[node] = '1'

                    if t == 2:
                        M.append((''.join(op), m[0][0][0][1],1))
                        break

    return M


Model = {}
for ar in Ar_All:
    pass

arr = ['11-110-0011-00101', '11-110-0011-00011']

for ar in Ar_All:
    SS = {}
    ar = ar.split('-')
    Ar_node = [(a,n) for n,a in enumerate(ar)]

    A = []
    for na,i in enumerate(ar):
        a = []
        a.append((i,na))
        if na > 0:
            for n1, j in enumerate(i[2:]):
                if j != '0':
                    a = a + A[n1]
            a = list(set(a))

        a = sorted(a, key=lambda stu: stu[1])
        A.append(a)

    # print(ar)
    # print(A)
    sp = search(Ar_node)
    # print(sp)
    # print(hebing(sp))

    sp = hebing(sp)
    # print(sp)
    model = [i[0] for i in sp]
    model = '-'.join(model)
    # print(model)
    if model not in Model:
        Model[model] = [ar]
    else:
        Model[model].append(ar)
# print('-'.join(model))

print(len(Ar_All),len(Model))

