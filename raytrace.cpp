//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

enum MaterialType {ROUGH, REFLECTIVE, REFRACTIVE};

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	float ior;
	MaterialType type;
	Material(MaterialType t) {type = t;}
};

struct RoughMaterial : Material
{
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH)
	{
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

struct ReflectiveMat : Material
{
	ReflectiveMat(vec3 n, vec3 kappa) : Material(REFLECTIVE)
	{
		/*
		kd = vec3(0.3f, 0.2f, 0.1f);
		ks = vec3(2, 2, 2);
		shininess = 50;
		*/
		vec3 one(1,1,1);
		F0 = ((n-one) *(n-one) + kappa * kappa) / ((n + one) * (n +one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Intersectable {
	vec3 param;

	Ellipsoid(const vec3& _param, Material* _material) {
		param = _param;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = -1;
		//my code
		float sx2 = ray.start.x * ray.start.x;
		float sy2 = ray.start.y * ray.start.y;
		float sz2 = ray.start.z * ray.start.z;
		float sdx = 2 * ray.start.x * ray.dir.x;
		float sdy = 2 * ray.start.y * ray.dir.y;
		float sdz = 2 * ray.start.z * ray.dir.z;
		float dx2 = ray.dir.x * ray.dir.x;
		float dy2 = ray.dir.y * ray.dir.y;
		float dz2 = ray.dir.z * ray.dir.z;

		float c = (param.x * sx2 + param.y * sy2 + param.z * sz2 - 1);
		float b = (param.x * sdx + param.y * sdy + param.z * sdz);
		float a = (param.x * dx2 + param.y * dy2 + param.z * dz2);
		//end of my code

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		//printf("T is: %f\n" ,hit.t);
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(hit.position * param);
		hit.material = material;
		return hit;
	}
};

//----------------------------
struct Plane : public Intersectable
{
	vec4 param;
	Material * reflectivematerial;
	vec3 points[5];
	float threshold;

	Plane(const vec3& _p1, const vec3& _p2, const vec3& _p3,const vec3& _p4, const vec3& _p5, const float& _threshold, Material * _roughmaterial, Material * _reflectivematerial)
	{
		threshold = _threshold;
		material = _roughmaterial;
		reflectivematerial = _reflectivematerial;
		points[0] = _p1;
		points[1] = _p2;
		points[2] = _p3;
		points[3] = _p4;
		points[4] = _p5;

		/*
		for (int i = 0; i< 5; ++i)
		{
			printf("Points are %f %f %f\n", points[i].x, points[i].y, points[i].z);
		}
		*/

		vec3 p1p2 = _p2 - _p1;
		vec3 p1p3 = _p3 - _p1;

		vec3 normal = cross(p1p2, p1p3);
		float d = -normal.x * _p1.x - normal.y * _p1.y - normal.z * _p1.z;

		param = vec4(normal.x, normal.y, normal.z, d);

		// printf("Parameters are: %f, %f, %f, %f\n", param.x, param.y, param.z, d);

	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;
		hit.t = -1;

		float sx = param.x * ray.start.x;
		float sy = param.y * ray.start.y;
		float sz = param.z * ray.start.z;

		float dx = param.x * ray.dir.x;
		float dy = param.y * ray.dir.y;
		float dz = param.z * ray.dir.z;

		hit.t = - (sx + sy + sz + param.w) / (dx + dy + dz);
		// printf("t is: %f\n", hit.t);
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(vec3(param.x, param.y, param.z));

		// Find nearest 2 points for edge distance calcuation
		vec3 nearestPoints[2] = {points[0], points[1]};
		float smallestLengths[2] = {length(hit.position - points[0]), length(hit.position - points[1])};
		for(int i = 2; i < 5; ++i)
		{
			vec3 newVector = hit.position - points[i];
			float lengthVector = length(newVector);
			if (lengthVector < smallestLengths[0] && lengthVector < smallestLengths[1])
			{
				if (smallestLengths[0] > smallestLengths[1])
				{
					nearestPoints[0] = points[i];
					smallestLengths[0] = lengthVector;
				}else
				{
					nearestPoints[1] = points[i];
					smallestLengths[1] = lengthVector;
				}
			}else if (lengthVector < smallestLengths[0])
			{
				nearestPoints[0] = points[i];
				smallestLengths[0] = lengthVector;
			}else if (lengthVector < smallestLengths[1])
			{
				nearestPoints[1] = points[i];
				smallestLengths[1] = lengthVector;
			}
		}
		// printf("Hit position: %f %f %f\n", hit.position.x, hit.position.y, hit.position.z);
		// printf("Closes 2 points are: %f %f %f,", nearestPoints[0].x, nearestPoints[0].y, nearestPoints[0].z);
		// printf(" %f %f %f\n", nearestPoints[1].x, nearestPoints[1].y, nearestPoints[1].z);

		// determine the distance to the closest edge
		vec3 ab = nearestPoints[1] - nearestPoints[0];
		vec3 ad = hit.position - nearestPoints[0];
		float cosTheta = dot(ab, ad) / (length(ab) * length(ad));
		float Theta = acos(cosTheta);
		float edgedistance = length(ad) * sin(Theta);

		if(edgedistance < threshold){hit.material = material;}
		else {hit.material = reflectivematerial;}
		return hit;
	}
};

struct Camera {
//---------------------------
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt)
	{
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, - d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye,lookat,up,fov);
	}
	/*
	void Animate(flfoat dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	*/
};

/*
struct Camera {
//---------------------------
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};
*/

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
	vec3 points[20] = {vec3 (0, 0.618, 1.618),
	vec3 (0, -0.618, 1.618),
	vec3 (0, -0.618, -1.618),
	vec3 (0, 0.618, -1.618),
	vec3 (1.618, 0, 0.618),
	vec3 (-1.618, 0, 0.618),
	vec3 (-1.618, 0, -0.618),
	vec3 (1.618, 0, -0.618),
	vec3 (0.618, 1.618, 0),
	vec3 (-0.618, 1.618, 0),
	vec3 (-0.618, -1.618, 0),
	vec3 (0.618, -1.618, 0),
	vec3 (1, 1, 1),
	vec3 (-1, 1, 1),
	vec3 (-1, -1, 1),
	vec3 (1, -1, 1),
	vec3 (1, -1, -1),
	vec3 (1, 1, -1),
	vec3 (-1, 1, -1),
	vec3 (-1, -1, -1)};

	int faces[60] = {1,2,16,5,13,
	1,13,9,10,14,
	1,14,6,15,2,
	2,15,11,12,16,
	3,4,18,8,17,
	3,17,12,11,20,
	3,20,7,19,4,
	19,10,9,18,4,
	16,12,17,8,5,
	5,8,18,9,13,
	14,10,19,7,6,
	6,7,20,11,15};

public:
	void build() {
		// printf("Start\n");
		vec3 eye = vec3(0, 0, 1.0f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.7f, 0.7f, 0.7f);
		vec3 lightDirection(1, 1, 1), Le(3, 3, 3);
		// lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material * material = new RoughMaterial(kd, ks, 50);

		vec3 kd2(0.1f, 0.2f, 0.3f), ks2(2, 2, 2);
		Material * material2 = new RoughMaterial(kd2, ks2, 50);

		vec3 nGold (0.17f, 0.35f, 1.5f);
		vec3 kGold (3.1f, 2.7f, 1.9f);
		ReflectiveMat* goldMaterial= new ReflectiveMat(nGold, kGold);

		vec3 nSilver (0.14f, 0.16f, 0.13f);
		vec3 kSilver (4.1f, 2.3f, 3.1f);
		ReflectiveMat* silverMaterial = new ReflectiveMat(nSilver, kSilver);


		int facesSize = sizeof(faces)/ sizeof(int);
		for (int i = 0; i < facesSize; i = i + 5)
		{
			objects.push_back(new Plane(points[faces[i]-1], points[faces[i+1]-1],
				 points[faces[i+2]-1], points[faces[i+3]-1], points[faces[i+4]-1],0.1f, material2, silverMaterial));
 			// objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));
		}
		// objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, goldMaterial));
		// objects.push_back(new Ellipsoid(vec3(1.3f,2.2f,1.9f), material));
		objects.push_back(new Ellipsoid(vec3(20.0f, 35.0f,40.0f), goldMaterial));
		// objects.push_back(new Plane(vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), material));
		// objects.push_back(new Plane(vec3(0,0,0), vec3(0,0,1), vec3(0,1,0), material));
		// objects.push_back(new Ellipsoid(vec3(.05f,.06f,.07f), goldMaterial));
		// objects.push_back(new Plane(vec3(-1.0f,2.0f,1.0f), vec3(0.0f,-3.0f,2.0f), vec3(1.0f,1.0f,-4.0f), material));
		// objects.push_back(new Plane(vec3(-2.0f,1.0f,-1.0f), vec3(0.0f,-2.0f,0.0f), vec3(1.0f,-1.0f,2.0f), material));
		// objects.push_back(new Sphere(vec3(0.0f, 0.0f, 0.0f), 0.2f, goldMaterial));
		// printf("End\n");
			// printf("random is %f\n", rnd());
	}

	void render(std::vector<vec4>& image) {
		// printf("Start\n");
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
		// printf("Start2\n");
			for (int X = 0; X < windowWidth; X++) {
				// printf("Start getray\n");
				vec3 color = trace(camera.getRay(X, Y));
				// printf("End getray\n");
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
			// printf("End2\n");
		}
		// printf("End\n");
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}


	vec3 trace(Ray ray, int depth = 0)
	{
	// printf("Start tracing\n");
	if (depth > 5) return La;
	Hit hit = firstIntersect(ray);
	if (hit.t <0) return La;
	if (hit.material->type == ROUGH)
	{
		// printf("Start rough\n");
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		// printf("End\n");
		return outRadiance;
	}
	// printf("Start reflective\n");
	float cosa = -dot(ray.dir, hit.normal);
	vec3 one(1,1,1);
	vec3 F = hit.material-> F0 + (one - hit.material->F0) * pow(1-cosa, 5);
	vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
	vec3 outRadiance = trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth +1) * F;

	if (hit.material -> type == REFRACTIVE)
	{
		// printf("Start refractive\n");
		float disc = 1 - (1- cosa * cosa) / hit.material-> ior / hit.material-> ior; // scalar n
		if (disc >= 0)
		{
			vec3 refractedDir = ray.dir / hit.material-> ior + hit.normal * (cosa / hit.material-> ior - sqrt(disc));
			outRadiance = outRadiance + trace(Ray(hit.position - hit.normal * epsilon, refractedDir), depth + 1) * (one- F);
		}
	}
	// printf("End\n");
	return outRadiance;
}
/*
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
	*/

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

//my code
/*
// fragment shader in GLSL
const char * const DodevertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const DodefragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";
*/

/*
class Dodecahedron
{
	unsigned int vao;	   // virtual world on the GPU
	public :
	Dodecahedron (float *vertices)
	{
	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed
		DodegpuProgram.create(DodevertexSource, DodefragmentSource, "outColor");
	}
void Draw()
{
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { .1, 0, 0, 0,    // MVP matrix,
								0, .1, 0, 0,    // row-major!
								0, 0, .1, 0,
								0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLE_STRIP, 0 , 20);

	glutSwapBuffers(); // exchange buffers for double buffering
	}
};
*/


class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %ld milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	printf("Key pressed\n");
	scene.Animate(0.1f);
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//printf("Animating");

}
