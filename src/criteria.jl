struct SimuInfo
    simulate_nums::Int  #模拟次数
    iteration::Int #现在的模拟次序，在模拟时要实时更新
end

struct MaxUCB
    c::Float64
    beta::Float64
end

function select_best(crit::MaxUCB, h_node::POWTreeObsNode,info::SimuInfo, rng) #UCT
    tree = h_node.tree
    h = h_node.node
    best_criterion_val = -Inf
    local best_node::Int
    istied = false
    local tied::Vector{Int}
    ltn = log(tree.total_n[h])
    for node in tree.tried[h] #观察的子动作结点
        n = tree.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = tree.v[node][length(tree.v[node])]
        elseif n == 0 && tree.v[node][length(tree.v[node])] == -Inf
            criterion_value = Inf
        else
            # print(info.iteration)
            history_value=crit.beta*sqrt(Range(tree.v[node])*Standard_Deviation(tree.v[node]))
            # println("这是第$(info.iteration)次模拟")
            if info.iteration <= info.simulate_nums/2 -1
                alpha=info.simulate_nums/(2*(info.iteration+1))*crit.c
            else 
                alpha=crit.c
            end
            criterion_value = tree.v[node][length(tree.v[node])]+ alpha*sqrt(ltn/n) + history_value
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
            istied = false
        elseif criterion_value == best_criterion_val
            if istied
                push!(tied, node)
            else
                istied = true
                tied = [best_node, node]
            end
        end
    end
    if istied
        return rand(rng, tied)
    else
        return best_node
    end
end

struct MaxQ end

function select_best(crit::MaxQ, h_node::POWTreeObsNode,info::SimuInfo, rng) #选择Q值最大的
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_v = tree.v[best_node][length(tree.v[best_node])]
    @assert !isnan(best_v)
    for node in tree.tried[h][2:end]
        if tree.v[node][length(tree.v[node])] >= best_v
            best_v = tree.v[node][length(tree.v[node])]
            best_node = node
        end
    end
    return best_node
end

struct MaxTries end

function select_best(crit::MaxTries, h_node::POWTreeObsNode,info::SimuInfo, rng) #选择访问次数最多的
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_n = tree.n[best_node]
    @assert !isnan(best_n)
    for node in tree.tried[h][2:end]
        if tree.n[node] >= best_n
            best_n = tree.n[node]
            best_node = node
        end
    end
    return best_node
end

function Range(A::Vector{Float64})
    max = A[1]
    min = A[1]
    for i in A
        if max < i
            max = i
        end
        if min > i
            min = i
        end
    end       
    max-min 
end

function Standard_Deviation(A::Vector{Float64})
    sum = 0
    for i in A
        sum+=i
    end
    mean = sum/length(A)
    result = 0
    for i in A
        result+=(i-mean)*(i-mean)    
    end
    result/=length(A)
    sqrt(result)
end

