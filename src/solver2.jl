function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}
    tree = h_node.tree
    h = h_node.node
    sol = pomcp.solver
    dep = 1.0
    # if POMDPs.isterminal(pomcp.problem, s)
    #     return 0.0, dep
    # end
    # if d <= 0
    #     return 0.0, dep
    # end
    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0, dep
    end

    # 选动作
    if sol.enable_action_pw #采用动作空间的渐进拓宽
        # 给观察结点增加一个动作
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action * total_n ^ sol.alpha_action
        # if length(tree.tried[h]) < 1
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), POWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))  #一个动作可以被重复抽取 或 这个观察结点没有对应的动作
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions 给观察结点增加全部动作
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    # 根据准则选择拓宽树的最优动作结点（UCT）
    best_node = select_best(pomcp.criterion, h_node,pomcp.history_info, pomcp.solver.rng)
    a = tree.a_labels[best_node]


    # 选观察
    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)
    # if tree.n_a_children[best_node] < 1

        sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o)) #检查重复的观察结点并动作结点有对应的观察子节点
            hao = tree.a_child_lookup[(best_node, o)]# 用已有的
        else
            # 只有新的观察结点才把观察保留
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
    else
        sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node  #创建的新观察，使用rollout方法
        dep = dep + 1.0
        R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
    else #对于已经存在的观察或观察结点数已经最大的情况下抽样加权的观察
        pair = rand(sol.rng, tree.generated[best_node]) #抽样一组观察
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])
        reward, depth = simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
        dep = depth + 1
        # dep = max(depth + 1, dep)
        R = r + POMDPs.discount(pomcp.problem) * reward
        # R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
    end

    tree.n[best_node] += 1 #访问动作结点次数+1
    tree.total_n[h] += 1 #访问观察结点次数+1
    Q=tree.v[best_node][length(tree.v[best_node])]
    if Q != -Inf
        push!(tree.v[best_node],Q+(R-Q)/tree.n[best_node])
    end

    return R, dep
end

