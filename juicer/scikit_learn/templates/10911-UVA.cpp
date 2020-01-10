#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <math.h>

using nampespace std;

float sum_of_distance = 9999999;

typedef pair<int, float> neighbor;

struct node {
	int x, y;
	bool marked;
	vector<neighbor> adj_list;
}

float distance(node &p1, node &p2){
	float pow1 = static_cast<float>(p1.x-p2.x);
	float pow2 = static_cast<float>(p1.y-p2.y);
	return sqrt(pow2+pow1);
}

struct graph {
	vector<node> nodes;
	
	void create_complete_graph(){
		int size_ = nodes.size();
		for(int i=0; i<size_; i++){
			for(int j=0; j<size_; j++){
				if(i!=j){
					neighbor n;
					n.first = j;
					n.second = distance(nodes[i], nodes[j]);
					nodes[i].adj_list.push_back(n);
				}
			}
		}
	}
}


int main(){
	int N, N2;
	string S;
	int x, y;

	cin >> N;

	while(N!=0){
		N2 = N*2;
		graph g;
		for(int i=0; i<N2; i++){
			cin >> S >> x >> y;
			node n;
			n.x = x;
			n.y = y;
			pair.marked = false;
			graph.nodes.push_back(n);
		}

		cin >> N;
	}


	return 0;
}