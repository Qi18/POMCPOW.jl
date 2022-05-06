struct POMCPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Int}  #这个动作点的访问次数
    v::Vector{Vector{Float64}}  #这个动作点的历史值估计
    generated::Vector{Vector{Pair{O,Int}}} #动作结点的子观察节点对indice的映射，在solver中相当于W使用
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A} #真正的动作
    n_a_children::Vector{Int} #动作结点的子观察结点数量

    # observation nodes
    sr_beliefs::Vector{B} # first element is #undef #对应的信念状态
    total_n::Vector{Int} # 这个观察点的访问次数
    max_v::Vector{Float64} # 这个观察点的值（最大动作值）
    tried::Vector{Vector{Int}} # 观察点的子节点序列
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params
    o_labels::Vector{O} #真正的观察

    # root
    root_belief::RB

    function POMCPOWTree{B,A,O,RB}(root_belief, sz::Int=1000) where{B,A,O,RB}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Int[], sz),
            sizehint!(Vector{Float64}[], sz),
            sizehint!(Vector{Pair{O,Int}}[], sz),
            Dict{Tuple{Int,O}, Int}(),
            sizehint!(A[], sz),
            sizehint!(Int[], sz),

            sizehint!(Array{B}(undef, 1), sz),
            sizehint!(Int[0], sz),
            sizehint!(Float64[-Inf], sz),
            sizehint!(Vector{Int}[Int[]], sz),
            Dict{Tuple{Int,A}, Int}(),
            sizehint!(Array{O}(undef, 1), sz),

            root_belief
        )
    end
end

@inline function push_anode!(tree::POMCPOWTree{B,A,O}, h::Int, a::A, n::Int=0, v::Float64=0.0, update_lookup=true) where {B,A,O} #给观察插入一个动作
    anode = length(tree.n) + 1
    # println("创建一个动作结点，现在它的序号是$(anode)")
    push!(tree.n, n)
    push!(tree.v, [v])
    # for i in tree.v
    #     println(i)
    #     println(length(i))
    # end
    # println("$(length([v]))")
    # println("$(tree.v[anode])")
    push!(tree.generated, Pair{O,Int}[])
    push!(tree.a_labels, a)
    push!(tree.n_a_children, 0)
    if update_lookup
        tree.o_child_lookup[(h, a)] = anode
    end
    # println(h)
    # println(tree.max_v[h])
    if tree.max_v[h] < v
        tree.max_v[h] = v
    end
    push!(tree.tried[h], anode)
    tree.total_n[h] += n
    return anode
end

struct POWTreeObsNode{B,A,O,RB} <: BeliefNode
    tree::POMCPOWTree{B,A,O,RB}
    node::Int
end

isroot(h::POWTreeObsNode) = h.node==1
@inline function belief(h::POWTreeObsNode)
    if isroot(h)
        return h.tree.root_belief
    else
        return StateBelief(h.tree.sr_beliefs[h.node])
    end
end
function sr_belief(h::POWTreeObsNode)
    if isroot(h)
        error("Tried to access the sr_belief for the root node in a POMCPOW tree")
    else
        return h.tree.sr_beliefs[h.node]
    end
end
n_children(h::POWTreeObsNode) = length(h.tree.tried[h.node])
