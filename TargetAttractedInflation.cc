#include "TargetAttractedInflation.hh"

#include <MeshFEM/ParallelAssembly.hh>

TargetAttractedInflation::TargetAttractedInflation(std::shared_ptr<InflatableSheet> s, const Mesh &targetSurface, const std::vector<bool>& wallVtxOnBoundary)
        : nondimensionalization(*s), m_sheet(s), m_targetSurfaceFitter(std::make_unique<TargetSurfaceFitter>(targetSurface))
{
    const auto &m = mesh();
    const size_t nv = m.numVertices();

    // Area in the original (unoptimized) rest mesh. This is used for scaling the
    // vertex contributions to the fitting energy.
    VXd vertexArea(VXd::Zero(nv));
    if (s->using_single_fusing_line()) {
        // if using single fusing lines, use the sum of 1/2 of incident fused edges' lengths as the area of the vertex.
        for (auto he : s->mesh().halfEdges()) {
            size_t v1 = he.tip().index();
            size_t v2 = he.tail().index();
            if (s->isWallVtx(v1) && s->isWallVtx(v2)) {
                vertexArea(v1) += 0.5 * (he.tip().node()->p - he.tail().node()->p).norm();
                vertexArea(v2) += 0.5 * (he.tip().node()->p - he.tail().node()->p).norm();
            }
        }
    } else {
        for (const auto &e : m.elements())
            for (const auto &v : e.vertices())
                vertexArea[v.index()] += e->volume() / 3.0;
    }

    const auto &wallVtxs = sheet().wallVertices();
    if (wallVtxOnBoundary.size() != wallVtxs.size()) {
        m_wallVtxOnBoundary.resize(wallVtxs.size());
        for (size_t i = 0; i < wallVtxs.size(); ++i) {
            m_wallVtxOnBoundary[i] = m.vertex(wallVtxs[i]).isBoundary();
        }
    } else {
        std::cout<<"Using provided wallVtxOnBoundary\n";
        // There could be internal holes that are not boundary features but rather in fusing patterns. We don't want boundary attraction forces for those vertices because these holes are not in the target surface.
        m_wallVtxOnBoundary = wallVtxOnBoundary;
    }
    VXd queryPtArea(wallVtxs.size());
    for (size_t i = 0; i < wallVtxs.size(); ++i)
        queryPtArea[i] = vertexArea[wallVtxs[i]];

    m_targetSurfaceFitter->setQueryPtWeights(queryPtArea);

    m_targetSurfaceFitter->updateClosestPoints(sheet().deformedWallVertexPositions(), m_wallVtxOnBoundary);
    m_targetSurfaceFitter->holdClosestPointsFixed = true;
}

TargetAttractedInflation::Real TargetAttractedInflation::energy(EnergyType etype) const {
    Real result = 0.0;
    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation))
        result += nondimensionalization.potentialEnergyScale() * m_sheet->energy();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting))
        result += (fittingWeight * nondimensionalization.fittingEnergyScale()) * m_targetSurfaceFitter->energy();

    return result;
}

TargetAttractedInflation::VXd TargetAttractedInflation::gradient(EnergyType etype) const {
    VXd g = VXd::Zero(numVars());

    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation)) {
        g += (nondimensionalization.potentialEnergyScale()
                    * nondimensionalization.equilibriumVarScale()) // Compute gradient with respect to our re-scaled equilibrium variables.
                    * m_sheet->gradient();
    }

    // Note: gradUnweightedTargetFit already computes the gradient with respect to the re-scaled equilibrium variables.
    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting))
        g += (fittingWeight * nondimensionalization.fittingEnergyScale()) * gradUnweightedTargetFit();

    return g;
}

TargetAttractedInflation::VXd TargetAttractedInflation::gradUnweightedTargetFit() const {
    const auto &s = sheet();

    VXd g = VXd::Zero(numVars());
    auto gradQueryPt = m_targetSurfaceFitter->gradient();
    gradQueryPt *= nondimensionalization.equilibriumVarScale(); // Compute gradient with respect to our re-scaled equilibrium variables.
    const auto &wv = s.wallVertices();
    for (size_t i = 0; i < wv.size(); ++i)
        g.segment<3>(s.varIdx(0, wv[i], 0)) += gradQueryPt.row(i).transpose();
    return g;
}

SuiteSparseMatrix TargetAttractedInflation::hessianSparsityPattern(Real val) const {
    return sheet().hessianSparsityPattern(val);
}

SuiteSparseMatrix TargetAttractedInflation::hessian(EnergyType etype) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype);
    return H;
}

void TargetAttractedInflation::hessian(SuiteSparseMatrix &H, EnergyType etype) const {
    const auto &s = sheet();

    H.setZero();
    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation))
        s.hessian(H);
    H *= nondimensionalization.potentialEnergyScale();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
        const auto &tsf = targetSurfaceFitter();
        const auto &wv = s.wallVertices();
        for (size_t i = 0; i < wv.size(); ++i) {
            const auto &vtxHess = ((fittingWeight * nondimensionalization.fittingEnergyScale()) * tsf.vtx_hessian(i)).eval();
            const size_t varOffset = s.varIdx(0, wv[i], 0);

            size_t hint = std::numeric_limits<int>::max(); // not size_t max which would be come -1 on cast to int!
            for (size_t comp_a = 0; comp_a < 3; ++comp_a)
                for (size_t comp_b = comp_a; comp_b < 3; ++comp_b)
                    hint = H.addNZ(varOffset + comp_a, varOffset + comp_b, vtxHess(comp_a, comp_b), hint);
        }
    }
    // Compute Hessian with respect to our re-scaled equilibrium variables.
    H *= std::pow(nondimensionalization.equilibriumVarScale(), 2);
}
