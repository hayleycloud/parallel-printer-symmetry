#pragma once

// CGAL headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>
#include <CGAL/AABB_halfedge_graph_segment_primitive.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polygon_mesh_slicer.h>
#include <CGAL/intersections.h>

//
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/centroid.h>
#include <CGAL/enum.h>

#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/nearest_neighbor_delaunay_2.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>

namespace PMP = CGAL::Polygon_mesh_processing;


#include <memory>
#include <queue>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K1;
typedef CGAL::Surface_mesh<K1::Point_3> Mesh;
typedef std::vector<K1::Point_3> Polyline_type;
typedef std::list<Polyline_type> Polylines;
typedef CGAL::AABB_halfedge_graph_segment_primitive<Mesh> HGSP;
typedef CGAL::AABB_traits<K1, HGSP> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

typedef std::pair<K1::Point_3, K1::Vector_3> HardPlane;

struct SubdivisionMeshNode
{
	int id;
	Mesh data;
	SubdivisionMeshNode* parent;
	std::vector<std::unique_ptr<SubdivisionMeshNode>> children;
};

typedef SubdivisionMeshNode SubdivisionTree;

struct SubdivisionOption;

struct SubdivisionOptionsNode
{
	SubdivisionOption* parent;
	Mesh mesh;
	std::vector<std::unique_ptr<SubdivisionOption>> options;
};

struct SubdivisionOption
{
	SubdivisionOptionsNode* parent;
	HardPlane subdivPlane;
	std::unique_ptr<SubdivisionOptionsNode> mesh1, mesh2;
};

typedef SubdivisionOptionsNode SubdivisionOptionsTree;

typedef std::vector<SubdivisionMeshNode*> PrinterArrayState;
typedef std::vector<SubdivisionOptionsNode*> PrinterOptionsArrayState;

struct LLNormal;

typedef std::pair<LLNormal,double> NormalFitnessPair;

class ParallelPrinter
{
public:
	ParallelPrinter(const std::string& inputFile);

	~ParallelPrinter() {};

private:
	void printPlane(const K1::Point_3& centroid, const K1::Vector_3& normal);

	void findSymmetries(const Mesh& mesh, std::vector<HardPlane>* planes);

	void findSymmetries(
		const Mesh& mesh,
		K1::Point_3* centroid,
		std::vector<NormalFitnessPair>* symmetries);

	PrinterArrayState subdivide(
		SubdivisionMeshNode* parent, PrinterArrayState printers, int ply);

	PrinterArrayState subdivide(PrinterArrayState printersState, int ply);

	std::unique_ptr<SubdivisionTree>
	explore(SubdivisionOptionsNode* mesh, PrinterArrayState* printerArray);

	bool explore(SubdivisionOptionsNode* parent, int depth);

	std::unique_ptr<SubdivisionOptionsNode> createOptionTreeFrom(
		SubdivisionMeshNode* parent, int depth);

	bool createOptionsFrom(SubdivisionOptionsNode* parent, int depth);

	void exploreOptionLeft(
		SubdivisionOptionsNode* mesh,
		std::queue<SubdivisionOption*>* pivots,
		PrinterOptionsArrayState printerArrayState,
		std::vector<PrinterOptionsArrayState>* printerArrayStates);

	void exploreOptionRight(
		SubdivisionOptionsNode* mesh,
		std::queue<SubdivisionOption*>* pivots,
		PrinterOptionsArrayState printerArrayState,
		std::vector<PrinterOptionsArrayState>* printerArrayStates);

	void explorePivots(
		std::queue<SubdivisionOption*>* pivots,
		PrinterOptionsArrayState printerArrayState,
		std::vector<PrinterOptionsArrayState>* printerArrayStates);

	void exploreOption(
		SubdivisionOption* option,
		std::queue<SubdivisionOption*>* pivots,
		PrinterOptionsArrayState printerArrayState,
		std::vector<PrinterOptionsArrayState>* printerArrayStates);

	void exportSubdivisionNodes(const std::vector<SubdivisionMeshNode*>& nodes);

	void exportIntermediaryNodes(const SubdivisionMeshNode& node);

	void exportSubdivisionTreeInstructions(const SubdivisionTree& tree);

	void decompose();

	void decompose(Mesh& mesh);
};
