#include "ParallelPrinter.h"

#include <omp.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

const double pi = 3.14159265359;

std::random_device randDev;
std::mt19937 randEng(randDev());

enum ReportingMode
{
	Quiet, Normal, Verbose
};

struct Arguments
{
	std::string inputFile;
	std::string outputDir;
	int numPrinters;
	int searchDepth;
	int numSamples;
	int iterations;
	double symmetryTolerance;
	std::optional<double> minVolume;
	double maxDimSize;
	double maxDimSizeSquared;
	ReportingMode reporting;
} args;

///////////////////////////////////////////////////////////////////////////////
// Utility Functions

void normalizeSpherical(double* variate)
{
	if(*variate < 0.0)
		*variate += 1.0;
	else if(*variate > 1.0)
		*variate -= 1.0;
}

const K1::Vector_3 normalize(const K1::Vector_3& v)
{
	double l = sqrt(v.squared_length());
	return K1::Vector_3(v.x() / l, v.y() / l, v.z() / l);
}

double dot(const K1::Vector_3& a, const K1::Vector_3& b)
{
	return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}


inline const std::pair<K1::Point_3, K1::Vector_3> hardPlane(
	const K1::Point_3& centroid, const K1::Vector_3& normal)
{
	return std::make_pair(centroid, normal);
}


///////////////////////////////////////////////////////////////////////////////
// CGAL Mesh Operations

bool validate(Mesh& mesh, bool throwOnFail)
{
	if(!CGAL::Polygon_mesh_processing::triangulate_faces(mesh))
	{
		if(throwOnFail)
			throw std::runtime_error("Mesh not triangulated!");
	}

	CGAL::Polygon_mesh_processing::autorefine(mesh);

	if(CGAL::Polygon_mesh_processing::does_self_intersect(mesh))
	{
		if(throwOnFail)
			throw std::runtime_error("Mesh self-intersects!");
		return false;
	}

	return true;
}

void loadSTLMesh(const std::string& filename, Mesh* cgalMesh)
{
	if(!PMP::IO::read_polygon_mesh(filename, *cgalMesh))
		throw std::runtime_error(filename + " could not be loaded!");
}


inline K1::Point_3 getCentroid(const Mesh& mesh)
{
	const Mesh::Property_map<Mesh::Vertex_index,K1::Point_3>& pnts = mesh.points();
	return CGAL::centroid(pnts.begin(), pnts.end(), CGAL::Dimension_tag<0>());
}

int leastSquaresFit(
	const Mesh& mesh,
	const K1::Point_3* centroid,
	K1::Line_3* line,
	double* length)
{
	std::vector<K1::Point_3> points;
	points.reserve(mesh.num_vertices());

	linear_least_squares_fitting_3(
		mesh.points().begin(), mesh.points().end(),
		*line,
		CGAL::Dimension_tag<0>());

	K1::Point_3 localCentroid;
	if(centroid == nullptr)
		localCentroid = getCentroid(mesh);
	else
		localCentroid = *centroid;

	double dMax = 0;
	for(const K1::Point_3& p: mesh.points())
	{
		K1::Vector_3 dv = localCentroid - p;
		double d = dv.squared_length();

		if(d > dMax)
			dMax = d;
	}

	*length = sqrt(dMax);

	return 0;
}

bool slice(const Mesh& inMesh, const HardPlane& p, Mesh* outMesh)
{
	*outMesh = Mesh(inMesh);

	bool result = CGAL::Polygon_mesh_processing::clip(
		*outMesh, 
		K1::Plane_3(p.first, p.second), 
		CGAL::Polygon_mesh_processing::parameters::clip_volume(true));

	result &= validate(*outMesh, false);

	return result;
}


///////////////////////////////////////////////////////////////////////////////
// Symmetry Determination

inline bool positivelyOrientedWithPlane(
	const HardPlane& symmetryPlane, const K1::Point_3& v)
{
    K1::Point_3 u = symmetryPlane.first;
	K1::Vector_3 dv = 
        K1::Vector_3(v.x(), v.y(), v.z()) - 
        K1::Vector_3(u.x(), u.y(), u.z());
	return dot(dv, symmetryPlane.second) > 0.0;
}

inline double distanceToPlane(const K1::Point_3& v, const HardPlane& plane)
{
    K1::Point_3 u = plane.first;
	K1::Vector_3 n = plane.second;
	K1::Vector_3 dv = 
        K1::Vector_3(v.x(), v.y(), v.z()) - 
        K1::Vector_3(u.x(), u.y(), u.z());
	
	return dot(dv, n);
}

inline void reflectInPlane(
	const K1::Point_3& v, const HardPlane& plane, K1::Point_3* vOut)
{
	K1::Point_3 w = v - (plane.second * distanceToPlane(v, plane) * 2.0);
	*vOut = w;
}

inline double meanAbsoluteError(const std::vector<double>& errors)
{
	double sumErrors = 0.0;
	for(auto error: errors)
		sumErrors += error;
	return sumErrors / static_cast<double>(errors.size());
}

typedef CGAL::Search_traits_3<K1> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree Tree;

double symmetryFitness(
	const Mesh& cgalMesh,
	const HardPlane& symmetryPlane,
	double planeThickness)
{
	std::vector<K1::Point_3> pointsSide1, pointsSide2;
	std::vector<K1::Point_3> sampleSide1, sampleSide2;
	for(K1::Point_3 vertex: cgalMesh.points())
	{
		K1::Point_3 reflPoint;
		double distToPlane = distanceToPlane(vertex, symmetryPlane); 
		reflectInPlane(vertex, symmetryPlane, &reflPoint);

		if(positivelyOrientedWithPlane(symmetryPlane, vertex))
		{
			pointsSide1.push_back(vertex);
			//if(distToPlane < planeThickness)
			//{
			//	pointsSide2.push_back(vertex);
			//	sampleSide1.push_back(reflPoint);
			//}
			sampleSide2.push_back(reflPoint);
		}
		else
		{
			//if(distToPlane < planeThickness)
			//{
			//	pointsSide1.push_back(vertex);
			//	sampleSide2.push_back(reflPoint);
			//}
			pointsSide2.push_back(vertex);
			sampleSide1.push_back(reflPoint);
		}
	}

	// Determine if sizes and dimensions are valid here
	/*for(const K1::Point_3& vertex1: sampleSide1)
	{
		for(const K1::Point_3& vertex2: sampleSide1)
		{
			if((vertex1 - vertex2).squared_length() > args.maxDimSizeSquared)
				return std::numeric_limits<double>::max();
		}
	}
	for(const K1::Point_3& vertex1: sampleSide2)
	{
		for(const K1::Point_3& vertex2: sampleSide2)
		{
			if((vertex1 - vertex2).squared_length() > args.maxDimSizeSquared)
				return std::numeric_limits<double>::max();
		}
	}*/

    Tree side1Tree(pointsSide1.begin(), pointsSide1.end());
    Tree side2Tree(pointsSide2.begin(), pointsSide2.end());
	std::vector<double> errors;

	//std::cout << "Processing side 1..." << std::endl;
	for(const K1::Point_3& vertex: sampleSide1)
	{
        Neighbor_search search(side1Tree, vertex, 1);
		//std::cout << "Processed vertex: " << vertex.x() << "," << vertex.y() << "," << vertex.z() << "." <<  std::endl;
		//std::cout << search.begin()->first << " " << std::sqrt(search.begin()->second) << std::endl;
		errors.push_back(std::sqrt(search.begin()->second));
	}

	//std::cout << "Processing side 2..." << std::endl;
	for(const K1::Point_3& vertex: sampleSide2)
	{
        Neighbor_search search(side2Tree, vertex, 1);
		//std::cout << "Processed vertex: " << vertex.x() << "," << vertex.y() << "," << vertex.z() << "." <<  std::endl;
		//std::cout << search.begin()->first << " " << std::sqrt(search.begin()->second) << std::endl;
		errors.push_back(std::sqrt(search.begin()->second));
	}

	return meanAbsoluteError(errors);
}

struct LLNormal
{
	double u, v;
	double theta, phi;
	K1::Vector_3 n;

	LLNormal(double u, double v)
		: u(u), v(v)
	{
		theta = std::acos((2.0 * v) - 1.0);
		phi = 2.0 * pi * u;

		n = K1::Vector_3(
			std::cos(phi) * std::sin(theta),
			std::sin(phi) * std::sin(theta),
			std::cos(theta));
	}

	double len() const
	{
		return std::sqrt(n.x() * n.x() + n.y() * n.y() + n.z() * n.z());
	}

	double angleBetween(const LLNormal& other) const
	{
		return std::acos(n.x() * other.n.x() + n.y() * other.n.y() + n.z() * other.n.z());
	}
};

void stochasticNormalSample(
	std::vector<LLNormal>* normals, 
	unsigned int sampleSize,
	std::uniform_real_distribution<double>& uRange,
	std::uniform_real_distribution<double>& vRange)
{
	for(unsigned int i = 0; i < sampleSize; ++i)
	{
		double u = uRange(randEng);
		double v = vRange(randEng);

		normalizeSpherical(&u);
		normalizeSpherical(&v);

		//std::cout << "Lat: " << lat << ", Long: " << lng << std::endl;
		normals->push_back(LLNormal(u, v));
	}
}

void condense(std::list<NormalFitnessPair>* normals, double arcLength)
{
	for(auto n1itr = normals->begin(); n1itr != normals->end(); ++n1itr)
	{
		NormalFitnessPair normal1 = *n1itr;
		for(auto n2itr = n1itr; n2itr != normals->end(); ++n2itr)
		{
			NormalFitnessPair normal2 = *n2itr;
			if(n2itr != n1itr)
			{
				double a = normal1.first.angleBetween(normal2.first);
				if(a <= arcLength || a >= pi - arcLength)
				{
					n2itr = normals->erase(n2itr);
					--n2itr;
				}
			}
		}
	}
}

void ParallelPrinter::findSymmetries(
	const Mesh& mesh, 
	K1::Point_3* centroid, 
	std::vector<NormalFitnessPair>* symmetries)
{
	// Determine planes of symmetry using a stochastic method
	const unsigned int iterations = args.iterations;
	const unsigned int sampleSize = args.numSamples;
	const unsigned int filterSize = 5;
	const double hardErrorTolerance = args.symmetryTolerance;
	double searchRadius = 0.1;
	K1::Point_3 centroidLocal = getCentroid(mesh);
	std::vector<LLNormal> bestNormals;
	std::vector<NormalFitnessPair> normals;

	for(unsigned int i = 1; i <= iterations; ++i)
	{
		std::vector<LLNormal> sampleNormals;

		if(bestNormals.size() == 0)
		{
			std::uniform_real_distribution<double>
				uRange(0.0, 1.0), vRange(0.0, 1.0);
			stochasticNormalSample(&sampleNormals, sampleSize, uRange, vRange);
		}
		else
		{
			const double halfSearchRad = searchRadius * 0.5;
			std::list<NormalFitnessPair> normalPairs;
			for(const LLNormal& normal: bestNormals)
			{
				const double u = normal.u, v = normal.v;
				std::uniform_real_distribution<double>
					uRange(u - halfSearchRad, u + halfSearchRad), 
					vRange(v - halfSearchRad, v + halfSearchRad);
				stochasticNormalSample(
					&sampleNormals, sampleSize / 3, uRange, vRange);
			}
		}

		std::list<NormalFitnessPair> eligibleNormals;

		#pragma omp parallel for
		for(const LLNormal& normal: sampleNormals)
		{
			HardPlane plane = hardPlane(centroidLocal, normal.n);
			double f = symmetryFitness(mesh, plane, 0.01);
			#pragma omp critical
			eligibleNormals.push_back(std::make_pair(normal, f));
		}

		eligibleNormals.sort([=](const NormalFitnessPair& op1, const NormalFitnessPair& op2){
			return op1.second < op2.second;
		});

		condense(&eligibleNormals, 0.5);

		//unsigned int j = 0;
		//for(const NormalFitnessPair& nPair: eligibleNormals)
		//{
		//	std::cout << i << "," << j << ": " << nPair.first.n << " [" << nPair.first.theta << ", " << nPair.first.phi << "] (" << nPair.second << ") " << std::endl;
		//	++j;
		//}

		std::list<NormalFitnessPair>::iterator nItr = eligibleNormals.begin();
		if(i == iterations)
		{
			for(unsigned int nIndex = 0; nIndex < filterSize; ++nIndex)
			{
				if((*nItr).second < hardErrorTolerance)
					normals.push_back(*nItr);
				if((++nItr) == eligibleNormals.end())
					break;
			}
		}
		else
		{
			symmetries->clear();
			for(unsigned int nIndex = 0; nIndex < filterSize; ++nIndex)
			{
				symmetries->push_back(*nItr);
				if((++nItr) == eligibleNormals.end())
					break;
			}
		}
	}

	*centroid = centroidLocal;
}

void ParallelPrinter::findSymmetries(
	const Mesh& mesh, std::vector<HardPlane>* planes)
{
	K1::Point_3 centroid;
	std::vector<NormalFitnessPair> normals;

	findSymmetries(mesh, &centroid, &normals);

	for(const auto& n: normals)
		planes->push_back(hardPlane(centroid, n.first.n));
}

///////////////////////////////////////////////////////////////////////////////
// 

double calculateCost(const Mesh& mesh)
{
	return CGAL::Polygon_mesh_processing::volume(mesh);
}

void createPrinterArrayState(int number, PrinterArrayState* printerArrayState)
{
	printerArrayState->resize(number);
	std::fill(printerArrayState->begin(), printerArrayState->end(), nullptr);
}

void createPrinterArrayState(int number, PrinterOptionsArrayState* printerArrayState)
{
	printerArrayState->resize(number);
	std::fill(printerArrayState->begin(), printerArrayState->end(), nullptr);
}

int addToPrinterArrayState(
	SubdivisionMeshNode* mesh,
	PrinterArrayState* printerArrayState,
	int startIndex = 0)
{
	for(int i = startIndex; i < printerArrayState->size(); ++i)
	{
		SubdivisionMeshNode* node = (*printerArrayState)[i];
		if(node == nullptr)
		{
			(*printerArrayState)[i] = mesh;
			return i;
		}
	}

	return -1;
}

int addToPrinterArrayState(
	SubdivisionOptionsNode* mesh,
	PrinterOptionsArrayState* printerArrayState,
	int startIndex = 0)
{
	for(int i = startIndex; i < printerArrayState->size(); ++i)
	{
		SubdivisionOptionsNode* node = (*printerArrayState)[i];
		if(node == nullptr)
		{
			(*printerArrayState)[i] = mesh;
			return i;
		}
	}

	return -1;
}

int expandIntoPrinterState(
	SubdivisionMeshNode* mesh, PrinterArrayState* printerArrayState)
{
	for(int i = 0; i < printerArrayState->size(); ++i)
	{
		if((*printerArrayState)[i] == mesh->parent)
		{
			assert(mesh->children.size() == 2);
			int nextIndex = addToPrinterArrayState(
				mesh->children[1].get(), printerArrayState, i);
			if(nextIndex < 0)
				return -1;

			(*printerArrayState)[i] = mesh->children[0].get();
			return nextIndex;
		}
	}

	return -1;
}

int expandIntoPrinterState(
	SubdivisionOption* node, PrinterOptionsArrayState* printerArrayState)
{
	for(int i = 0; i < printerArrayState->size(); ++i)
	{
		if((*printerArrayState)[i] == node->parent)
		{
			int nextIndex = addToPrinterArrayState(
				node->mesh2.get(), printerArrayState, i);
			if(nextIndex < 0)
				return -1;

			(*printerArrayState)[i] = node->mesh1.get();
			return nextIndex;
		}
	}

	return -1;
}

bool isPrinterArrayFull(const PrinterArrayState& printersState)
{
	for(SubdivisionMeshNode* node: printersState)
	{
		if(node == nullptr)
			return false;
	}

	return true;
}

bool isPrinterArrayFull(const PrinterOptionsArrayState& printersState)
{
	for(SubdivisionOptionsNode* node: printersState)
	{
		if(node == nullptr)
			return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Heuristic-Based Subdivision Logic

struct SymmetryScore
{
	SubdivisionMeshNode* node;
	HardPlane symmetry;
	double score;
};

PrinterArrayState ParallelPrinter::subdivide(
	PrinterArrayState printersState, int ply)
{
	if(isPrinterArrayFull(printersState))
	{
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "No more printers available." << std::endl;
		return printersState;
	}

	//std::vector<PrinterArrayState
	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Subdivision." << std::endl;
		
	size_t numPrinters = printersState.size();

	std::list<SymmetryScore> symmetries;
	// Iterate through the subdivisions so far (in the state)
#pragma omp parallel for
	for(unsigned int i = 0; i < numPrinters; ++i)
	{
		SubdivisionMeshNode* node = printersState[i];
		if(node == nullptr)
			continue;

		validate(node->data, false);

		K1::Point_3 centroid;
		std::vector<NormalFitnessPair> normals;
		findSymmetries(node->data, &centroid, &normals);

		NormalFitnessPair& bestNormal = normals[0];
#pragma omp critical
		symmetries.push_back({
			node, hardPlane(centroid, bestNormal.first.n), bestNormal.second
		});
		// Determine the "effectiveness score" of the symmetry based on proportion 
	}
	
	// Subdivide each mesh as appropriate and construct pathing options
	// (i.e. ignore ones we've determined are "final" to avoid reprocessing)
	// (we also designate any final meshes as "final" here)
	
	// We want to divide by the best symmetries, so sort them
	symmetries.sort([=](const SymmetryScore& op1, const SymmetryScore& op2){
		return op1.score > op2.score;
	});
	
	// and then go through them, best-first
	bool stagnated = true;	// Test for convergence
	for(const SymmetryScore& symmetry: symmetries)
	{
		//std::cout << symmetry.node << ": (C: ";
		//std::cout << symmetry.symmetry.first.x() << ", ";
		//std::cout << symmetry.symmetry.first.y() << ", ";
		//std::cout << symmetry.symmetry.first.z() << "; N: ";
		//std::cout << symmetry.symmetry.second.x() << ", ";
		//std::cout << symmetry.symmetry.second.y() << ", ";
		//std::cout << symmetry.symmetry.second.z() << ")";
		//std::cout << std::endl;

		// Find the printer associated with the symmetry
		for(unsigned int i = 0; i < numPrinters; ++i)
		{
			if(printersState[i] == symmetry.node)
			{
				unsigned int j = i;
				for(; j < numPrinters; ++j)
				{
					if(printersState[j] == nullptr)
						break;
				}
				if(j >= numPrinters)
				{
					if(args.reporting == ReportingMode::Verbose)
						std::cout << "No more printers available." << std::endl;
					return printersState;
				}

				std::unique_ptr<SubdivisionMeshNode> node1 =
					std::make_unique<SubdivisionMeshNode>();
				std::unique_ptr<SubdivisionMeshNode> node2 =
					std::make_unique<SubdivisionMeshNode>();

				bool sliceStatus =
					slice(symmetry.node->data, symmetry.symmetry, &node1->data);
				assert(sliceStatus);
				
				if(node1->data.number_of_vertices() <= 1)
				{
					if(args.reporting == ReportingMode::Verbose)
						std::cerr << "Node 1: Empty mesh detected." << std::endl;
					continue;
				}
				if(CGAL::Polygon_mesh_processing::volume(node1->data) <
				   args.minVolume)
				{
					std::cerr << "Node 1: Mesh below minimum constraint." << std::endl;
					continue;
				}

				HardPlane inverseSymmetry = hardPlane(
					symmetry.symmetry.first, -symmetry.symmetry.second);

				sliceStatus =
					slice(symmetry.node->data, inverseSymmetry, &node2->data);
				assert(sliceStatus);

				if(node2->data.number_of_vertices() <= 1)
				{
					if(args.reporting == ReportingMode::Verbose)
						std::cerr << "Node 2: Empty mesh detected." << std::endl;
					continue;
				}
				if(CGAL::Polygon_mesh_processing::volume(node2->data) <
				   args.minVolume)
				{
					std::cerr << "Node 2: Mesh below minimum constraint." << std::endl;
					continue;
				}

				printersState[i] = node1.get();
				printersState[j] = node2.get();

				if(args.reporting == ReportingMode::Verbose)
					std::cout << i << " => " << i << " + " << j << std::endl;

				node1->id = -1;
				node2->id = -1;
				node1->parent = symmetry.node;
				node2->parent = symmetry.node;
				symmetry.node->children.push_back(std::move(node1));
				symmetry.node->children.push_back(std::move(node2));

				stagnated = false;
			}
		}
	}
	if(stagnated)
	{
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "No more symmetries detected." << std::endl;
		return printersState;
	}

	return subdivide(printersState, ply - 1);
}

PrinterArrayState ParallelPrinter::subdivide(
	SubdivisionMeshNode* parent, PrinterArrayState printers, int ply)
{
	printers[0] = parent;
	return subdivide(printers, ply);
}

///////////////////////////////////////////////////////////////////////////////
// Forward-Search Exploration Management

int count = 0;

bool ParallelPrinter::createOptionsFrom(SubdivisionOptionsNode* parent, int depth)
{
	assert(depth > 0);

	int nextDepth = depth - 1;
	if(nextDepth == 0)
	{
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "Reached final depth." << std::endl;
		return true;
	}

	std::vector<HardPlane> symmetries;
	findSymmetries(parent->mesh, &symmetries);

	if(symmetries.size() == 0) {
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "No symmetries found." << std::endl;
		return false;
	}

	++count;
	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Count: " << count << ", Depth: " << depth << ", Planes: " << symmetries.size() << ", Parent: " << parent << std::endl;

#pragma omp parallel for
	for(const HardPlane& symmetry: symmetries)
	{
		std::unique_ptr<SubdivisionOption> node = std::make_unique<SubdivisionOption>();
		node->parent = parent;
		node->subdivPlane = symmetry;

		std::unique_ptr<SubdivisionOptionsNode> node1 = std::make_unique<SubdivisionOptionsNode>();
		std::unique_ptr<SubdivisionOptionsNode> node2 = std::make_unique<SubdivisionOptionsNode>();
		node1->parent = node.get();
		node2->parent = node.get();

		HardPlane inverseSymmetry = hardPlane(symmetry.first, -symmetry.second);

		bool nodeComplete = true;
		if(slice(parent->mesh, symmetry, &node1->mesh))
		{
			if(CGAL::Polygon_mesh_processing::volume(node1->mesh) < args.minVolume)
				continue;

			if(createOptionsFrom(node1.get(), nextDepth))
				node->mesh1 = std::move(node1);
			else
				continue;
		}
		else
			continue;

		if(slice(parent->mesh, inverseSymmetry, &node2->mesh))
		{
			if(CGAL::Polygon_mesh_processing::volume(node2->mesh) < args.minVolume)
				continue;

			if(createOptionsFrom(node2.get(), nextDepth))
				node->mesh2 = std::move(node2);
			else
				continue;
		}
		else
			continue;

#pragma omp critical
		parent->options.push_back(std::move(node));
	}

	return true;
}

std::unique_ptr<SubdivisionOptionsNode>
ParallelPrinter::createOptionTreeFrom(SubdivisionMeshNode* parent, int depth)
{
	std::vector<HardPlane> symmetries;
	findSymmetries(parent->data, &symmetries);

	if(symmetries.size() == 0) {
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "No symmetries found." << std::endl;
		return nullptr;
	}

	++count;
	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Count: " << count << ", Depth: " << depth << ", Planes: " << symmetries.size() << ", Parent: " << parent << std::endl;

	std::unique_ptr<SubdivisionOptionsNode> parentNode = std::make_unique<SubdivisionOptionsNode>();
	parentNode->parent = nullptr;
	parentNode->mesh = parent->data;

#pragma omp parallel for
	for(const HardPlane& symmetry: symmetries)
	{
		std::unique_ptr<SubdivisionOption> node = std::make_unique<SubdivisionOption>();
		node->parent = parentNode.get();
		node->subdivPlane = symmetry;

		std::unique_ptr<SubdivisionOptionsNode> node1 = std::make_unique<SubdivisionOptionsNode>();
		std::unique_ptr<SubdivisionOptionsNode> node2 = std::make_unique<SubdivisionOptionsNode>();
		node1->parent = node.get();
		node2->parent = node.get();

		HardPlane inverseSymmetry = hardPlane(symmetry.first, -symmetry.second);

		bool nodeComplete = true;
		if(slice(parent->data, symmetry, &node1->mesh))
		{
			if(createOptionsFrom(node1.get(), depth - 1))
				node->mesh1 = std::move(node1);
			else
				continue;
		}
		else
			continue;

		if(slice(parent->data, inverseSymmetry, &node2->mesh))
		{
			if(createOptionsFrom(node2.get(), depth - 1))
				node->mesh2 = std::move(node2);
			else
				continue;
		}
		else
			continue;

#pragma omp critical
		parentNode->options.push_back(std::move(node));
	}

	return parentNode;
}

std::unique_ptr<SubdivisionTree> convertFromOpts(
	PrinterOptionsArrayState& optsArray, PrinterArrayState* outArray)
{
	std::unique_ptr<SubdivisionMeshNode> treeRoot = nullptr;
	std::unordered_map<SubdivisionOptionsNode*,SubdivisionMeshNode*> nodeMap;

	for(SubdivisionOptionsNode* optionsNode: optsArray)
	{
		if(optionsNode)
		{
			std::unique_ptr<SubdivisionMeshNode> leafNode =
				std::make_unique<SubdivisionMeshNode>();
			leafNode->id = -1;
			leafNode->data = optionsNode->mesh;
			leafNode->parent = nullptr;

			addToPrinterArrayState(leafNode.get(), outArray);

			nodeMap[optionsNode] = leafNode.get();
			std::unique_ptr<SubdivisionMeshNode> lastNode = std::move(leafNode);

			if(optionsNode->parent)
			{
				SubdivisionOptionsNode* nextNode = optionsNode->parent->parent;
				while(nextNode)
				{
					// We've bumped into somewhere we've been before, so merge
					if(nodeMap.find(nextNode) != nodeMap.end())
					{
						lastNode->parent = nodeMap[nextNode];
						nodeMap[nextNode]->children.push_back(std::move(lastNode));
						nextNode = nullptr;
					}
					// Add the next parent
					else
					{
						std::unique_ptr<SubdivisionMeshNode> node =
							std::make_unique<SubdivisionMeshNode>();
						node->id = -1;
						node->data = nextNode->mesh;
						node->parent = nullptr;

						nodeMap[nextNode] = node.get();
						lastNode->parent = node.get();
						node->children.push_back(std::move(lastNode));

						lastNode = std::move(node);
						if(nextNode->parent && optionsNode->parent->parent)
							nextNode = nextNode->parent->parent;
						else
						{
							// Found the root!
							// Root should not exist at this point
							// If it does, it should've been in the map!
							assert(treeRoot == nullptr);
							treeRoot = std::move(lastNode);
							nextNode = nullptr;
						}
					}
				}
			}
		}
	}

	return std::move(treeRoot);
}

bool ParallelPrinter::explore(SubdivisionOptionsNode* parent, int depth)
{
	return true;
}

std::unique_ptr<SubdivisionTree> ParallelPrinter::explore(
	SubdivisionOptionsNode* mesh, PrinterArrayState* printerArray)
{
	assert(mesh);
	assert(printerArray);

	std::vector<PrinterOptionsArrayState> printerOptArrays;
	PrinterOptionsArrayState defaultOptArrayState;
	std::fill(defaultOptArrayState.begin(), defaultOptArrayState.end(), nullptr);
	defaultOptArrayState.resize(printerArray->size());
	defaultOptArrayState[0] = mesh;

	for(auto& option: mesh->options)
	{
		std::queue<SubdivisionOption*> emptyPivots;
		exploreOption(
			option.get(),
			&emptyPivots,
			defaultOptArrayState, &printerOptArrays);
	}

	std::vector<double> costs;
	for(const PrinterOptionsArrayState& state: printerOptArrays)
	{
		double worstCost = 0.0;
		for(SubdivisionOptionsNode* node: state)
		{
			if(node != nullptr)
			{
				double cost;
				if(node->mesh.num_edges() <= 0
				   || node->mesh.num_vertices() <= 0)
					cost = std::numeric_limits<double>::max();
				else
					cost = calculateCost(node->mesh);

				if(cost > worstCost)
					worstCost = cost;
			}
		}
		if(worstCost > 0.0)
			costs.push_back(worstCost);
	}

	double bestCost = std::numeric_limits<double>::max();
	unsigned int bestCostIndex = std::numeric_limits<unsigned int>::max();
	unsigned int costIndex = 0;
	for(double cost: costs)
	{
		if(cost < bestCost)
		{
			bestCost = cost;
			bestCostIndex = costIndex;
		}
		++costIndex;
	}

	std::cout << "Exploration Results:" << std::endl;
	std::cout << "Printer Array States Found: " << printerOptArrays.size() << std::endl;
	std::cout << "Best (" << bestCost << "):" << std::endl;
	for(SubdivisionOptionsNode* printState: printerOptArrays[bestCostIndex])
	{
		if(printState != nullptr)
			std::cout << "(" << printState << ": " << calculateCost(printState->mesh) << ")" << std::endl;
		else
			std::cout << "[Empty]" << std::endl;
	}

	return convertFromOpts(printerOptArrays[bestCostIndex], printerArray);
}

void ParallelPrinter::exploreOptionLeft(
	SubdivisionOptionsNode* mesh,
	std::queue<SubdivisionOption*>* pivots,
	PrinterOptionsArrayState printerArrayState,
	std::vector<PrinterOptionsArrayState>* printerArrayStates)
{
	assert(mesh);
	assert(pivots);
	assert(printerArrayStates);

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Exploring left..." << std::endl;

	for(auto& subOption: mesh->options)
	{
		exploreOption(
			subOption.get(), pivots, printerArrayState, printerArrayStates);
	}

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Done exploring left." << std::endl;
}

void ParallelPrinter::exploreOptionRight(
	SubdivisionOptionsNode* mesh,
	std::queue<SubdivisionOption*>* pivots,
	PrinterOptionsArrayState printerArrayState,
	std::vector<PrinterOptionsArrayState>* printerArrayStates)
{
	assert(mesh);
	assert(pivots);
	assert(printerArrayStates);

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Explore right..." << std::endl;

	for(auto& subOption: mesh->options)
	{
		exploreOption(
			subOption.get(), pivots, printerArrayState, printerArrayStates);
	}

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Done exploring right." << std::endl;
}

void ParallelPrinter::explorePivots(
	std::queue<SubdivisionOption*>* pivots,
	PrinterOptionsArrayState printerArrayState,
	std::vector<PrinterOptionsArrayState>* printerArrayStates)
{
	std::queue<SubdivisionOption*> pivotsCpy(*pivots);

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Exploring " << pivots->size() << " pivots..." << std::endl;

	while(!pivotsCpy.empty())
	{
		SubdivisionOption* pivot = pivotsCpy.front();
		pivotsCpy.pop();

		std::queue<SubdivisionOption*> newQueue;
		exploreOptionRight(
			pivot->mesh2.get(),
			&newQueue,
			printerArrayState, printerArrayStates);
	}

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Done exploring pivots." << std::endl;
}

void ParallelPrinter::exploreOption(
	SubdivisionOption* option,
	std::queue<SubdivisionOption*>* pivots,
	PrinterOptionsArrayState printerArrayState,
	std::vector<PrinterOptionsArrayState>* printerArrayStates)
{
	assert(option);
	assert(pivots);
	assert(printerArrayStates);

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Exploring option..." << std::endl;

	if(expandIntoPrinterState(option, &printerArrayState) < 0)
	{
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "Expanding into state failed." << std::endl;
		return;
	}
	if(args.reporting == ReportingMode::Verbose)
	{
		for(auto& opt: printerArrayState)
				std::cout << opt << std::endl;
		std::cout << std::endl;
	}
	printerArrayStates->push_back(printerArrayState);

	explorePivots(pivots, printerArrayState, printerArrayStates);

	exploreOptionRight(
		option->mesh2.get(), pivots, printerArrayState, printerArrayStates);

	std::queue<SubdivisionOption*> pivots2(*pivots);
	pivots2.push(option);
	exploreOptionLeft(
		option->mesh1.get(), &pivots2, printerArrayState, printerArrayStates);

	if(args.reporting == ReportingMode::Verbose)
		std::cout << "Done exploring option." << std::endl;
}

inline void indentation(int num)
{
	for(int i = 0; i < num; ++i)
		std::cout << "    ";
}

void printSubdivisionNode(const SubdivisionMeshNode* node, int depth = 0)
{
	if(depth == 0)
	{
		if(args.reporting != ReportingMode::Quiet)
			std::cout << "Subdivision Tree:" << std::endl;
		assert(node->data.number_of_vertices() > 1);
		assert(CGAL::is_closed(node->data));
		double area = CGAL::Polygon_mesh_processing::area(node->data);
		double vol = CGAL::Polygon_mesh_processing::volume(node->data);
		if(args.reporting != ReportingMode::Quiet)
			std::cout << "Base Mesh [" << node->id << "]: (Area: " << area << ", Volume: " << vol << ")" << std::endl;
	}

	for(const auto& subNode: node->children)
	{
		double area = CGAL::Polygon_mesh_processing::area(subNode->data);
		double vol = CGAL::Polygon_mesh_processing::volume(subNode->data);

		if(args.reporting != ReportingMode::Quiet)
		{
			indentation(depth + 1);
			std::cout << "Sub Mesh [" << subNode->id << "]: (Area: " << area << ", Volume: " << vol << ")" << std::endl;
		}
		printSubdivisionNode(subNode.get(), depth + 1);
	}
}

void printSubdivisionOptsNode(const SubdivisionOptionsNode* node, int depth = 0)
{
	if(depth == 0)
	{
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "Subdivision Options Tree:" << std::endl;
		assert(node->mesh.number_of_vertices() > 1);
		assert(CGAL::is_closed(node->mesh));
		double area = CGAL::Polygon_mesh_processing::area(node->mesh);
		double vol = CGAL::Polygon_mesh_processing::volume(node->mesh);
		if(args.reporting == ReportingMode::Verbose)
			std::cout << "Base Mesh: (Area: " << area << ", Volume: " << vol << ")" << std::endl;
	}

	for(const auto& subNode: node->options)
	{
		SubdivisionOptionsNode* child1 = subNode->mesh1.get();
		SubdivisionOptionsNode* child2 = subNode->mesh2.get();
		double area1 = CGAL::Polygon_mesh_processing::area(child1->mesh);
		double area2 = CGAL::Polygon_mesh_processing::area(child2->mesh);
		double vol1 = CGAL::Polygon_mesh_processing::volume(child1->mesh);
		double vol2 = CGAL::Polygon_mesh_processing::volume(child2->mesh);

		if(args.reporting == ReportingMode::Verbose)
		{
			indentation(depth + 1);
			std::cout << "Subdivision Pair:" << std::endl;
		}

		if(args.reporting == ReportingMode::Verbose)
		{
			indentation(depth + 1);
			std::cout << "Sub Mesh #1: (Area: " << area1 << ", Volume: " << vol1 << ")" << std::endl;
		}
		printSubdivisionOptsNode(child1, depth + 1);

		if(args.reporting == ReportingMode::Verbose)
		{
			indentation(depth + 1);
			std::cout << "Sub Mesh #2: (Area: " << area2 << ", Volume: " << vol2 << ")" << std::endl;
		}
		printSubdivisionOptsNode(child2, depth + 1);
	}
}

void findLeaves(
	SubdivisionMeshNode* tree, std::vector<SubdivisionMeshNode*>* leaves)
{
	std::vector<SubdivisionMeshNode*> layer;
	layer.push_back(tree);

	while(!layer.empty())
	{
		std::vector<SubdivisionMeshNode*> nextLayer;
		for(SubdivisionMeshNode* subNode: layer)
		{
			if(subNode->children.size() == 0)
				leaves->push_back(subNode);
			else
			{
				for(auto& child: subNode->children)
					nextLayer.push_back(child.get());
			}
		}

		layer = nextLayer;
	}
}

void flattenTree(
	const SubdivisionTree& tree,
	std::vector<std::vector<const SubdivisionMeshNode*>>* layers)
{
	std::vector<const SubdivisionMeshNode*> treeLayer;
	treeLayer.push_back(&tree);
	layers->push_back(treeLayer);

	bool notLastLayer = true;
	while(notLastLayer)
	{
		notLastLayer = false;

		std::vector<const SubdivisionMeshNode*> layer;
		for(const SubdivisionMeshNode* node: (*layers).back())
		{
			for(auto& child: node->children)
			{
				notLastLayer = true;
				layer.push_back(child.get());
			}
		}

		if(notLastLayer)
			layers->push_back(layer);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Exporting Operations

void determineIDs(std::vector<SubdivisionMeshNode*>* leaves)
{
	int id = 0;
	for(SubdivisionMeshNode* node: *leaves)
		if(node)
			node->id = ++id;

	std::vector<SubdivisionMeshNode*> layer(*leaves);
	while(!layer.empty())
	{
		std::set<SubdivisionMeshNode*> parentLayer;
		for(SubdivisionMeshNode* node: layer)
			if(node)
				if(node->parent)
					parentLayer.insert(node->parent);

		for(SubdivisionMeshNode* node: parentLayer)
			if(node->id < 0)
				node->id = ++id;

		layer = std::vector<SubdivisionMeshNode*>(parentLayer.begin(), parentLayer.end());
	}
}

void ParallelPrinter::exportSubdivisionNodes(const std::vector<SubdivisionMeshNode*>& nodes)
{
	for(SubdivisionMeshNode* node: nodes)
	{
		if(!node)
			continue;

		std::stringstream ss("");
		ss << node->id << ".stl";
        CGAL::IO::write_STL("split_out/" + ss.str(), node->data);
	}
}

void ParallelPrinter::exportIntermediaryNodes(const SubdivisionMeshNode& node)
{
	for(const auto& child: node.children)
	{
		std::stringstream ss("");
		ss << node.id << ".stl";
        CGAL::IO::write_STL("split_out/inters/" + ss.str(), node.data);

		exportIntermediaryNodes(*child);
	}
}

void ParallelPrinter::exportSubdivisionTreeInstructions(const SubdivisionTree& tree)
{
	std::stack<std::string> instructions;

	std::vector<std::vector<const SubdivisionMeshNode*>> layers;
	flattenTree(tree, &layers);

	for(auto layerItr = layers.rbegin(); layerItr != layers.rend(); ++layerItr)
	{
		auto& layer = *layerItr;
		for(unsigned int i = 0; i < layer.size(); ++i)
		{
			if((i+1) >= layer.size())
				continue;

			assert(layer[i]->parent == layer[i+1]->parent);

			std::stringstream instruction("");
			instruction << layer[i]->id << "+";
			++i;
			instruction << layer[i]->id;
			instruction << " => " << layer[i]->parent->id << std::endl;
			instructions.push(instruction.str());
		}
	}

	std::ofstream outFile("split_out/assembly.txt");
	while(!instructions.empty())
	{
		outFile << instructions.top();
		instructions.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////
// Base Operations

void ParallelPrinter::decompose(Mesh& mesh)
{
	std::cout << std::endl << "Decomposing Mesh..." << std::endl;
	std::cout << "-----------------------" << std::endl;

	SubdivisionTree tree;
	tree.parent = nullptr;
	tree.id = -1;
	tree.data = mesh;

	std::cout << "Creating options tree from mesh..." << std::endl;

	std::unique_ptr<SubdivisionOptionsTree> options =
		createOptionTreeFrom(&tree, args.searchDepth + 1);

	printSubdivisionOptsNode(options.get());

	std::cout << "Finding best decomposition configuration..." << std::endl;

	PrinterArrayState printers;
	createPrinterArrayState(args.numPrinters, &printers);

	std::unique_ptr<SubdivisionTree> finalTree =
		explore(options.get(), &printers);

	printSubdivisionNode(finalTree.get());

	std::cout << "Subdividing result..." << std::endl;
	printers = subdivide(printers, 3);

	// Prepare the directory for outputting mesh and assembly instructions
	try {
		std::filesystem::remove_all("split_out");
		std::filesystem::create_directory("split_out");
		std::filesystem::create_directory("split_out/inters");
	} catch(const std::exception& ex)
	{
		std::cerr << "Filesystem Exception Thrown: " << ex.what() << std::endl;
		return;
	}

	std::cout << "Assigning IDs..." << std::endl;
	determineIDs(&printers);
	printSubdivisionNode(static_cast<SubdivisionMeshNode*>(finalTree.get()));

	std::cout << "Exporting nodes as mesh files, plus assembly instructions." << std::endl;
	// Discard std cout
	std::ofstream hiddenStream("/dev/null");
	std::streambuf* coutRestore(std::cout.rdbuf());
	
	std::cout.rdbuf(hiddenStream.rdbuf());

	// Export the meshes and assembly instructions
	exportSubdivisionNodes(printers);
	exportIntermediaryNodes(*finalTree);
	exportSubdivisionTreeInstructions(*finalTree);

	// Restore std cout
	std::cout.rdbuf(coutRestore);

}

void ParallelPrinter::decompose()
{
	// deprecated
}


///////////////////////////////////////////////////////////////////////////////
// Lifetime 

ParallelPrinter::ParallelPrinter(const std::string& inputFile)
{
	std::srand(std::time(nullptr));

	std::chrono::time_point<std::chrono::steady_clock> startInterval =
		std::chrono::steady_clock::now();

	Mesh cgalMesh;
	loadSTLMesh(args.inputFile, &cgalMesh);
	validate(cgalMesh, true);

	if(!args.minVolume.has_value())
		args.minVolume =
			CGAL::Polygon_mesh_processing::volume(cgalMesh) /
			(args.numPrinters * 1.5);

	decompose(cgalMesh);

	std::chrono::time_point<std::chrono::steady_clock> finishInterval =
		std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> duration =
		finishInterval - startInterval;

	std::cout << "Computational Duration: "
	          << duration.count() << " ms." << std::endl;
}

int main(int argc, char** argv)
{
	// Default arguments
	const double maxDimSizeDefault = 200.0;
	args = {
		"",		// Input File
		"split_out",		// Output Directory
		0,		// Number of printers
		3,		// Search depth
		100,	// Number of samples
		3,		// Iterations
		1.0,	// Symmetry error tolerance
		std::optional<double>(),	// Minimum model size
		maxDimSizeDefault,
		maxDimSizeDefault * maxDimSizeDefault,
		ReportingMode::Normal
	};

	bool helpRequested = false;
	for(int argIndex = 0; argIndex < argc; ++argIndex)
	{
		std::string arg = argv[argIndex];
		if(arg == "-n" || arg == "--printers")
			args.numPrinters = std::stoi(argv[++argIndex]);
		if(arg == "-t" || arg == "--tolerance")
			args.symmetryTolerance = std::stod(argv[++argIndex]);
		else if(arg == "-s" || arg == "--samples")
			args.numSamples = std::stoi(argv[++argIndex]);
		else if(arg == "-m" || arg == "--min-size")
			args.minVolume = std::stoi(argv[++argIndex]);
		else if(arg == "-d" || arg == "--depth")
			args.searchDepth = std::stoi(argv[++argIndex]);
		else if(arg == "-i" || arg == "--iterations")
			args.iterations = std::stoi(argv[++argIndex]);
		else if(arg == "-v" || arg == "--verbose")
			args.reporting = ReportingMode::Verbose;
		else if(arg == "-q" || arg == "--quiet")
			args.reporting = ReportingMode::Quiet;
		else if(arg == "--o" || arg == "--output")
			args.outputDir = argv[++argIndex];
		else if(arg == "-h" || arg == "--help")
			helpRequested = true;
		else
			args.inputFile = argv[argIndex];
	}

	if(helpRequested || args.inputFile.length() <= 0 || args.numPrinters <= 1)
	{
		std::cout << "Usage: parallelprint [options] file" << std::endl;
		std::cout << "Options:" << std::endl;
		std::cout << "\t-n, --printers\tNumber of printers. [REQUIRED]" << std::endl;
		std::cout << "\t-d, --depth\tMaximum depth of the smart search phase." << std::endl;
		std::cout << "\t-h, --help\tShow this help screen." << std::endl;
		std::cout << "\t-i, --iters\tMaximum total iterations." << std::endl;
		std::cout << "\t-m, --min-size\tMinimum valid model size." << std::endl;
		std::cout << "\t-s, --samples\tNumber of symmetry sample points per iteration." << std::endl;
		std::cout << "\t-t, --tolerance\tSymmetry error tolerance." << std::endl;
		std::cout << "\t-v, --verbose\tVerbose state feedback reporting." << std::endl;
		std::cout << "\t-q, --quiet\tSuppress most state feedback reporting." << std::endl;
		std::cout << std::endl;
		std::cout << "Number of printers and an input file are required." << std::endl;

		return 0;
	}

	ParallelPrinter mainClass(args.inputFile);

	return 0;
}

