import math
import configparser
import random

import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
np.seterr(divide='raise')

plt.rcParams["figure.figsize"] = (16, 10)
class Asteroid:
    """Class containing all methods pertaining to asteroid gravity, feature computation and selection, and the gravity gradient matrix

    Args:
        asteroid_name: string containing the name of the asteroid, extracted from 'config.ini'
        title: string to be used in all plots containing the asteroid. Contains the full name of the asteroid
        file_location: string containing the file location for the selected polyhedral asteroid model
        asteroid_mass: mass of the asteroid, kg
        asteroid_ang_vel: angular velocity of the asteroid assumed entirely around the Z-axis, rad/sec
        verts: [n x 3] matrix of the vertice coordinates, km
        facets: [m x 3] matrix of the vertex indices that make up the tri, unitless
        edge_verts: [n+m-2 x 2] matrix containing the vertices of all UNIQUE edges, unitless
        edge_map: [2*(n+m-2) x 3] matrix. Columns 1 and 2 contain the vertices of every edge for each facet. This means
                  each edge is in here twice for tri-based polyhedral (once for each direction the edge can point).
                  Column 3 contains the corresponding facet index, unitless
        e_e_mat: [3 x 3 x n+m-2] matrix containing every unique edge's 3x3 E_E matrix
        edge_mat: [n+m-2] vector containing every unique edge's magnitude
        edge_vert_1: [n+m-2, 3] matrix containing the vector to the first vertex making up every unique edge
        edge_vert_2: [n+m-2 x 3] matrix containing the vector to the second vertex making up every unique edge
        f_f_mat: [3 x 3 x m] matrix containing every facet's 3x3 f_f matrix
        v_1: [m x 3]: matrix containing the vector to the first vertex making up the facet
        v_2: [m x 3]: matrix containing the vector to the second vertex making up the facet
        v_3: [m x 3]: matrix containing the vector to the third vertex making up the facet
        mu_grav_param: scalar for the gravitational parameter of the asteroid, m^3/s^2
        rho: scalar for the density of the asteroid, kg/m^3
        feature_verts: [num, 3] array containing the vectors to the feature vertices, units
        num_features: number of features on asteroid, unitless
    """
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read('input/' + config_file)
        self.asteroid_name = config.get('Asteroid', 'asteroid')  # string for name of the asteroid
        self.num_features = int(config.get('Asteroid', 'fnum'))  # number of features on asteroid

        self.title, self.file_location, self.asteroid_mass, self.asteroid_ang_vel = self.params(self.asteroid_name)
        self.verts, self.facets, self.edge_verts, self.edge_map = self.poly_setup()
        self.e_e_mat, self.edge_mat, self.edge_vert_1, self.edge_vert_2, self.f_f_mat, self.v_1, self.v_2, self.v_3 = self.setuppolygrav()
        self.mu_grav_param = 6.6742 * 10 ** -20 * self.asteroid_mass  # km^3/s^2
        self.rho = self.asteroid_mass / self.polyhedral_volume()
        self.feature_verts, self.feature_indices = self.featureverts()
        self.facet_centroids = self.verts[self.facets[:, 0:3]].mean(axis=1)
        self.facet_normals = self.get_facet_normals()

    def params(self, name):
        """Method to output the plot titles, file location, mass, and angular velocity of a specified asteroid

        Args:
            name: string containing the name of an asteroid with a model. Must be lowercase or it will error out

        Returns:
            title: string containing the title for use in plotting
            file_location: string containing the file location of the polyhedral shape model
            asteroid_mass: scalar mass of the asteroid, kg
            asteroid_ang_vel: scalar angular velocity of the asteroid around the Z-axis, rad/s
        """
        if name == "bennu":
            title = "101955 Bennu"
            file_location = "data/models/101955bennu.tab"
            asteroid_mass = 7.329 * 10 ** 10
            asteroid_ang_vel = 2 * math.pi / 4.296057 / 60 / 60
        elif name == "kleopatra":
            title = "216 Kleopatra"
            file_location = "data/models/216kleopatra.tab"
            asteroid_mass = 2.97 * 10 ** 18
            asteroid_ang_vel = 2 * math.pi / 5.385280 / 60 / 60
        elif name == "geographos":
            title = "1620 Geographos"
            file_location = "data/models/1620geographos.tab"
            asteroid_mass = 4 * 10 ** 12
            asteroid_ang_vel = 2 * math.pi / 5.223 / 60 / 60
        elif name == "bacchus":
            title = "2063 Bacchus"
            file_location = "data/models/2063bacchus.tab"
            asteroid_mass = 3.3 * 10 ** 12
            asteroid_ang_vel = 2 * math.pi / 14.544 / 60 / 60
        elif name == "toutatis":
            title = "4179 Toutatis"
            file_location = "data/models/4179toutatis.tab"
            asteroid_mass = 1.9 * 10 ** 13
            asteroid_ang_vel = 2 * math.pi / 176 / 60 / 60
        elif name == "toutatis2":
            title = "4179 Toutatis (hi-res)"
            file_location = "data/models/4179toutatis2.tab"
            asteroid_mass = 1.9 * 10 ** 13
            asteroid_ang_vel = 2 * math.pi / 176 / 60 / 60
        elif name == "castalia":
            title = "4769 Castalia"
            file_location = "data/models/4769castalia.tab"
            asteroid_mass = 5 * 10 ** 11
            asteroid_ang_vel = 2 * math.pi / 4.07 / 60 / 60
        elif name == "golevka":
            title = "6489 Golevka"
            file_location = "data/models/6489golevka.tab"
            asteroid_mass = 2.1 * 10 ** 11
            asteroid_ang_vel = 2 * math.pi / 6.026 / 60 / 60
        elif name == "itokawa":
            title = "25143 Itokawa"
            file_location = "data/models/25143itokawa.tab"
            asteroid_mass = 3.51 * 10 ** 10
            asteroid_ang_vel = 2 * math.pi / 12.132 / 60 / 60
        elif name == "apophis":
            title = "99942 Apophis"
            file_location = "data/models/Apophis_July22.obj"
            asteroid_mass = 4.5 * 10 ** 10
            asteroid_ang_vel = 2 * math.pi / 30.55 / 60 / 60
        else:
            print(
                "Either an invalid asteroid name has been provided, or the model exists, but doesn't have the necessary information available to run simulations on. See list of acceptable inputs:")
            print("bennu, kleopatra, geographos, bacchus, toutatis, toutatis2, castalia, golevka, itokawa, apophis")
            quit()
        return title, file_location, asteroid_mass, asteroid_ang_vel

    def poly_setup(self):
        """Method to generate the necessary vectors and arrays for computing its gravity and shape model given its file location

        Args:
            file_location: string containing the file location in the repo with the polyhedral shape model

        Returns:
            verts: [n x 3] matrix of the vertice coordinates, km
            facets: [m x 3] matrix of the vertex indices that make up the tri, unitless
            edge_verts: [n+m-2 x 2] matrix containing the vertices of all UNIQUE edges, unitless
            edge_map: [2*(n+m-2) x 3] matrix. Columns 1 and 2 contain the vertices of every edge for each facet. This means
                      each edge is in here twice for tri-based polyhedral (once for each direction the edge can point).
                      Column 3 contains the corresponding facet index, unitless
        """

        rawdata = np.loadtxt(self.file_location, dtype=str)

        numv = np.sum(np.where(rawdata == 'v', 1, 0))
        numf = np.sum(np.where(rawdata == 'f', 1, 0))

        verts = rawdata[0:numv, 1:4]
        facets = np.concatenate((rawdata[numv:numf + numv, 1:4], rawdata[numv:numf + numv, 1].reshape(numf, 1)), axis=1)
        verts = verts.astype(float)
        facets = np.add(facets.astype(int), -1)

        nume = numv + numf - 2
        edge_verts = np.zeros([nume * 2, 2])
        edge_map = np.zeros([nume * 2, 3])

        k = 0
        for i in range(numf):
            for j in range(facets.shape[1] - 1):
                edge_verts[k, :] = facets[i, j:j + 2]
                edge_map[k, :] = np.append(facets[i, j:j + 2], i)
                k += 1

        edge_verts = np.sort(edge_verts, axis=1)
        edge_verts = np.unique(edge_verts, axis=0)

        return verts, facets, edge_verts.astype(int), edge_map.astype(int)

    def polyhedral_volume(self):
        """Method computes the volume (units^3) of the polyhedron in the model. This is used for density computations

        Args:
            verts: [n x 3] matrix of the vertice coordinates, units
            facets: [m x 3] matrix of the vertex indices that make up the tri, unitless

        Returns:
            VOL: scalar value for the asteroid volume, units^3
        """

        verts = self.verts
        facets = self.facets

        m = facets.shape[0]
        vol = 0
        for i in range(m):
            v1 = verts[facets[i, 0], :]
            v2 = verts[facets[i, 1], :]
            v3 = verts[facets[i, 2], :]
            vol += np.dot(np.cross(v1, v2), v3)

        return vol/6

    def setuppolygrav(self):
        """populates the matrices and vectors of face and edge normals necessary for newpolygrav()

        Args:
            verts: [n x 3] matrix of the vertice coordinates, km
            facets: [m x 3] matrix of the vertex indices that make up the tri, unitless
            edge_verts: [n+m-2 x 2] matrix containing the vertices of all UNIQUE edges, unitless
            edge_map: [2*(n+m-2) x 3] matrix. Columns 1 and 2 contain the vertices of every edge for each facet. This means
                      each edge is in here twice for tri-based polyhedral (once for each direction the edge can point).
                      Column 3 contains the correspoinding facet index, unitless

        Returns:
            e_e_mat: [3 x 3 x n+m-2] matrix containing every unique edge's 3x3 E_E matrix
            edge_mat: [n+m-2] vector containing every unique edge's magnitude
            edge_vert_1: [n+m-2, 3] matrix containing the vector to the first vertex making up every unique edge
            edge_vert_2: [n+m-2 x 3] matrix containing the vector to the second vertex making up every unique edge
            f_f_mat: [3 x 3 x m] matrix containing every facet's 3x3 f_f matrix
            v_1: [m x 3]: matrix containing the vector to the first vertex making up the facet
            v_2: [m x 3]: matrix containing the vector to the second vertex making up the facet
            v_3: [m x 3]: matrix containing the vector to the third vertex making up the facet
        """
        verts = self.verts
        facets = self.facets
        edge_verts = self.edge_verts
        edge_map = self.edge_map
        n = verts.shape[0]
        m = facets.shape[0]

        e_e_mat = np.zeros([3, 3, n + m - 2])
        edge_mat = np.zeros(n + m - 2)
        edge_vert_1 = np.zeros([n + m - 2, 3])
        edge_vert_2 = np.zeros([n + m - 2, 3])
        f_f_mat = np.zeros([3, 3, m])
        v_1 = np.zeros([m, 3])
        v_2 = np.zeros([m, 3])
        v_3 = np.zeros([m, 3])

        # edge terms
        for i in range(edge_verts.shape[0]):
            # compute what facets share the current edge (indexed by i)

            # indices where the vertices are in the current edge (check both directions of the edge)
            [findex1, _] = np.where((edge_verts[i, :] == edge_map[:, 0:2]))
            [findex2, _] = np.where((edge_verts[i, ::-1] == edge_map[:, 0:2]))

            # want both vertices in the edge, not just one of the two
            findex1, c1 = np.unique(findex1, return_counts=True)
            findex2, c2 = np.unique(findex2, return_counts=True)
            findex1 = int(findex1[c1 > 1])
            findex2 = int(findex2[c2 > 1])

            # extract two vectors from these facets to compute normals

            # face 1, vector 1
            ind11 = facets[edge_map[findex1, 2], 0:2]
            vec11 = verts[ind11[1], :] - verts[ind11[0], :]

            # face 1, vector 2
            ind12 = facets[edge_map[findex1, 2], 1:3]
            vec12 = verts[ind12[1], :] - verts[ind12[0], :]

            # face 2, vector 1
            ind21 = facets[edge_map[findex2, 2], 0:2]
            vec21 = verts[ind21[1], :] - verts[ind21[0], :]

            # face 2, vector 2
            ind22 = facets[edge_map[findex2, 2], 1:3]
            vec22 = verts[ind22[1], :] - verts[ind22[0], :]

            # normal for facet 1
            n1hat = np.cross(vec11, vec12)
            n1hat = n1hat / np.linalg.norm(n1hat)

            # normal for facet 2
            n2hat = np.cross(vec21, vec22)
            n2hat = n2hat / np.linalg.norm(n2hat)

            # edge vectors
            e_v1 = verts[edge_verts[i, 0], :]
            e_v2 = verts[edge_verts[i, 1], :]
            edge_vert_1[i, :] = e_v1
            edge_vert_2[i, :] = e_v2
            e1vec = e_v2 - e_v1
            e2vec = e_v1 - e_v2

            # edge normals
            ne1hat = np.cross(e1vec, n1hat)
            ne1hat = ne1hat / np.linalg.norm(ne1hat)

            ne2hat = np.cross(e2vec, n2hat)
            ne2hat = ne2hat / np.linalg.norm(ne2hat)

            # E_e
            E_e = np.matmul(n1hat.reshape(3, 1), ne1hat.reshape(1, 3)) + np.matmul(n2hat.reshape(3, 1),
                                                                                   ne2hat.reshape(1, 3))

            e_e_mat[:, :, i] = E_e

            # edge magnitude
            edge_mat[i] = np.linalg.norm(e1vec)

        # facet terms
        for i in range(facets.shape[0]):
            # extract 2 vectors from the facet to compute normals

            # facet 1, vector 1
            ind11 = facets[i, 0:2]
            vec11 = verts[ind11[1], :] - verts[ind11[0], :]

            # facet 1, vector 2
            ind12 = facets[i, 1:3]
            vec12 = verts[ind12[1], :] - verts[ind12[0], :]

            # normal for facet 1
            nfhat = np.cross(vec11, vec12)
            nfhat = nfhat / np.linalg.norm(nfhat)

            # f_f
            f_f_mat[:, :, i] = np.matmul(nfhat.reshape(3, 1), nfhat.reshape(1, 3))

            # w_f
            v_1[i, :] = verts[ind11[0], :]
            v_2[i, :] = verts[ind11[1], :]
            v_3[i, :] = verts[ind12[1], :]

        return e_e_mat, edge_mat, edge_vert_1, edge_vert_2, f_f_mat, v_1, v_2, v_3


    def matmul_3d(self, A, B):
        """Finds the matrix multiplication for each nth matrix of ixj and jxs

        Args:
            A: [i x j x n] matrix
            B: [j x s x n] matrix

        Returns:
            C: [i x s x n] matrix
        """
        #print(A.shape)
        #print(B.shape)
        #print(np.transpose(A, (2, 0, 1)).shape)
        #print(np.transpose(B, (2, 0, 1)).shape)
        #print('C')
        #print(np.matmul(np.transpose(A, (2, 0, 1)), np.transpose(B, (2, 0, 1))).shape)
        return np.transpose(np.matmul(np.transpose(A, (2, 0, 1)), np.transpose(B, (2, 0, 1))), (1, 2, 0))

    def pt_mass_grav(self, field_point):
        accel = -self.mu_grav_param / np.linalg.norm(field_point)**3 * field_point
        sa = 0
        return accel, sa

    def newpolygrav_vec(self, field_point):
        """Method computes the gravitational acceleration with respect to a polyhedral model of an asteroid

        Args:
            field_point: [1 x 3] vector for coords of the field point, km
            e_e_mat: [3 x 3 x n+m-2] matrix containing every unique edge's 3x3 E_E matrix
            edge_mat: [n+m-2] vector containing every unique edge's magnitude
            edge_vert_1: [3 x n+m-2] matrix containing the vector to the first vertex making up every unique edge
            edge_vert_2: [3 x n+m-2] matrix containing the vector to the second vertex making up every unique edge
            f_f_mat: [3 x 3 x m] matrix containing every facet's 3x3 f_f matrix
            v_1: [3 x m]: matrix containing the vector to the first vertex making up the facet
            v_2: [3 x m]: matrix containing the vector to the second vertex making up the facet
            v_3: [3 x m]: matrix containing the vector to the third vertex making up the facet
            rho: scalar, density of asteroid, kg/km^3

        Returns:
            accel: [1x3] vector of acceleration in x,y,z direction, km/s^2
            sa: [1] scalar value of the solid angle, steradians.
        """

        GRAV_CONST = 6.6742 * 10 ** -20

        # edge terms
        O_mat = np.repeat(field_point, self.edge_mat.size).reshape([3, 1, self.edge_mat.size])
        r_e = self.edge_vert_1[:, :, np.newaxis]
        r_e = np.moveaxis(r_e, [0, 2, 1], [2, 1, 0]) - O_mat
        r1 = np.squeeze(np.linalg.norm(r_e, axis=0))
        r2 = self.edge_vert_2[:, :, np.newaxis]
        r2 = np.squeeze(np.linalg.norm(np.moveaxis(r2, [0, 2, 1], [2, 1, 0]) - O_mat, axis=0))
        L_e = np.log((r1 + r2 + self.edge_mat) / (r1 + r2 - self.edge_mat)).reshape([1, 1, self.edge_mat.size])
        #print('start')
        #test = self.matmul_3d(self.e_e_mat, self.matmul_3d(r_e, L_e))
        #print('test shape')
        #print(test.shape)
        edgesum = np.sum(self.matmul_3d(self.e_e_mat, self.matmul_3d(r_e, L_e)), axis=2)

        # face terms
        O_mat = O_mat[:, :, :self.v_1.shape[0]]
        v_110 = self.v_1[:, :, np.newaxis]
        v_111 = self.v_2[:, :, np.newaxis]
        v_121 = self.v_3[:, :, np.newaxis]
        r1f = np.moveaxis(v_110, [0, 2, 1], [2, 1, 0]) - O_mat
        r2f = np.moveaxis(v_111, [0, 2, 1], [2, 1, 0]) - O_mat
        r3f = np.moveaxis(v_121, [0, 2, 1], [2, 1, 0]) - O_mat
        nr1f = np.squeeze(np.linalg.norm(r1f, axis=0))
        nr2f = np.squeeze(np.linalg.norm(r2f, axis=0))
        nr3f = np.squeeze(np.linalg.norm(r3f, axis=0))
        r1f = r1f.reshape([3, -1])
        r2f = r2f.reshape([3, -1])
        w_f = 2 * np.arctan2(np.sum(r1f * np.cross(r2f, r3f.reshape([3, -1]), axisa=0, axisb=0).T, axis=0),
                             nr1f * nr2f * nr3f + nr1f * np.sum(r2f * r3f.reshape([3, -1]), axis=0) + nr2f * np.sum(
                                 r3f.reshape([3, -1]) * r1f, axis=0) + nr3f * np.sum(r1f * r2f, axis=0))
        facetsum = np.sum(self.matmul_3d(self.f_f_mat, self.matmul_3d(r3f, w_f.reshape([1, 1, -1]))), axis=2)
        accel = GRAV_CONST * self.rho * (facetsum - edgesum)
        sa = np.sum(w_f)  # recall: sa = 4pi means inside asteroid, sa = 2pi = on asteroid surface, sa = 0 means outside
        # print('true solid angle')
        # print(sa)
        # print('approx solid angle')
        # print(2* np.sum(np.sum(r1f * np.cross(r2f, r3f.reshape([3, -1]), axisa=0, axisb=0).T, axis=0)/
        #             (nr1f * nr2f * nr3f + nr1f * np.sum(r2f * r3f.reshape([3, -1]), axis=0) + nr2f * np.sum(
        #                         r3f.reshape([3, -1]) * r1f, axis=0) + nr3f * np.sum(r1f * r2f, axis=0))))
        return accel, sa

    def gravgrad_vec(self, field_point):
        """computes the gravity gradient matrix at a specific field point with respect to a polyhedral model of an asteroid

        Args:
            field_point: [1 x 3] vector for coords of the field point, km
            e_e_mat: [3 x 3 x n+m-2] matrix containing every unique edge's 3x3 E_E matrix
            edge_mat: [n+m-2] vector containing every unique edge's magnitude
            edge_vert_1: [3 x n+m-2] matrix containing the vector to the first vertex making up every unique edge
            edge_vert_2: [3 x n+m-2] matrix containing the vector to the second vertex making up every unique edge
            f_f_mat: [3 x 3 x m] matrix containing every facet's 3x3 f_f matrix
            v_1: [3 x m]: matrix containing the vector to the first vertex making up the facet
            v_2: [3 x m]: matrix containing the vector to the second vertex making up the facet
            v_3: [3 x m]: matrix containing the vector to the third vertex making up the facet
            rho: scalar, density of asteroid, kg/km^3

        Returns:
            gg: [3x3] matrix of gravity gradient, 1/s^2
        """

        GRAV_CONST = 6.6742 * 10 ** -20

        # edge terms
        O_mat = np.repeat(field_point,self.edge_mat.size).reshape([3,1,self.edge_mat.size])
        r_e = self.edge_vert_1[:, :, np.newaxis]
        r_e = np.moveaxis(r_e, [0, 2, 1], [2, 1, 0]) - O_mat
        r1 = np.squeeze(np.linalg.norm(r_e, axis=0))
        r2 = self.edge_vert_2[:, :, np.newaxis]
        r2 = np.squeeze(np.linalg.norm(np.moveaxis(r2, [0, 2, 1], [2, 1, 0]) - O_mat, axis=0))
        L_e = np.log((r1 + r2 + self.edge_mat) / (r1 + r2 - self.edge_mat))

        edgesum = np.sum(self.e_e_mat*L_e, axis=2)

        # facet terms
        O_mat = O_mat[:, :, :self.v_1.shape[0]]
        v_110 = self.v_1[:, :, np.newaxis]
        v_111 = self.v_2[:, :, np.newaxis]
        v_121 = self.v_3[:, :, np.newaxis]
        r1f = np.moveaxis(v_110, [0, 2, 1], [2, 1, 0]) - O_mat
        r2f = np.moveaxis(v_111, [0, 2, 1], [2, 1, 0]) - O_mat
        r3f = np.moveaxis(v_121, [0, 2, 1], [2, 1, 0]) - O_mat
        nr1f = np.squeeze(np.linalg.norm(r1f, axis=0))
        nr2f = np.squeeze(np.linalg.norm(r2f, axis=0))
        nr3f = np.squeeze(np.linalg.norm(r3f, axis=0))
        r1f = r1f.reshape([3, -1])
        r2f = r2f.reshape([3, -1])
        w_f = 2 * np.arctan2(np.sum(r1f*np.cross(r2f, r3f.reshape([3, -1]), axisa=0, axisb=0).T, axis=0),
                             nr1f * nr2f * nr3f + nr1f * np.sum(r2f*r3f.reshape([3, -1]), axis=0) + nr2f * np.sum(
                                 r3f.reshape([3, -1])*r1f, axis=0) + nr3f * np.sum(r1f*r2f, axis=0))
        facetsum = np.sum(self.f_f_mat*w_f, axis=2)

        gg = GRAV_CONST * self.rho * (edgesum - facetsum)
        return gg

    def pointfind(self, r_d, points):
        """Method returns the points closest to r_d within a given set of vertices (shape model)

        Args:
            points: [n x 3] matrix of the vertice coordinates, km
            r_d: [3] vector containing x-y-z point of interest, km

        Returns:
            points[ind, :]: [3] vector containing closest points in the shape model, km
            ind: [scalar] index of which number vertex is the closest, unitless
        """

        r_d_mat = np.tile(r_d, points.shape[0]).reshape([points.shape[0], 3])
        vertdist = np.linalg.norm(r_d_mat - points, axis=1)
        ind = int(np.argmin(vertdist))
        return points[ind, :], ind

    def featureverts(self):
        """Method returns the vertex indices in [verts] of the [num] closest vertices evenly spaced around [r_d] at [dist] distance away

        Args:
            verts: [n x 3] matrix of the vertice coordinates, units
            num_features: [1] scalar containing how many features (vertices) the user wants, unitless

        Returns:
            feature_verts: [num, 3] array containing the vectors to the feature vertices, units
        """
        verts = np.copy(self.verts)
        num_features = np.copy(self.num_features)
        ind_list = np.arange(0, verts.shape[0], 1, dtype=int)

        # initialize feature verts array
        feature_verts = np.zeros([num_features, 3])
        feature_indices = np.zeros(num_features, dtype=int)

        # select the rest of the features
        for i in range(num_features):
            ind = random.randint(0,len(verts)-1)
            feature_verts[i, :] = verts[ind, :]
            feature_indices[i] = ind_list[ind]
            verts = np.delete(verts, ind, axis=0)
            ind_list = np.delete(ind_list, ind, axis=0)

        return feature_verts, feature_indices


    def plotpoly(self):
        """Method outputs the figure and axis objects from matplotlib for a 3d asteroid shape model plot

        Args:
            verts: [n x 3] matrix of the vertice coordinates, units
            facets: [m x 3] matrix of the vertex indices that make up the tri, unitless
            title: string containing the title for use in plotting

        Returns:
            fig: matplotlib figure object containing the figure information
            ax: matplotlib axis object containing the 3d shape model plotted within fig
        """
        verts = self.verts
        facets = self.facets
        title = self.title

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        xverts = verts[:, 0]
        yverts = verts[:, 1]
        zverts = verts[:, 2]
        tris = facets[:, 0:3]
        ax.plot_trisurf(xverts, yverts, zverts, triangles=tris, cmap='viridis')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title)
        #ax.set_aspect('equal')
        return fig, ax

    def get_facet_normals(self):
        """Method computes the outward-facing facet normals for every facet in the polyhedral shape model

        Returns:
            normals: [n x 3] matrix of surface normals centered at facet centroids
        """
        v0 = self.verts[self.facets[:, 0]]
        v1 = self.verts[self.facets[:, 1]]
        v2 = self.verts[self.facets[:, 2]]
        normals = np.cross(v1-v0, v2-v0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return normals



def main():
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    print(asteroid.mu_grav_param)
    #print(asteroid.edge_verts)
    accel, sa = asteroid.newpolygrav_vec(np.array([0.3,0.3,0.3]))
    print(accel)
    print(sa)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(1000*asteroid.verts[:,0], 1000*asteroid.verts[:,1], 1000*asteroid.verts[:,2], marker='.',alpha=0.2, color='blue')
    ax.scatter(1000*asteroid.feature_verts[:,0], 1000*asteroid.feature_verts[:,1], 1000*asteroid.feature_verts[:,2], alpha=0.8, marker='o', color='red')
    ax.quiver(1000*asteroid.facet_centroids[:,0], 1000*asteroid.facet_centroids[:,1], 1000*asteroid.facet_centroids[:,2], 10*asteroid.facet_normals[:,0], 10*asteroid.facet_normals[:,1], 10*asteroid.facet_normals[:,2])
    ax.legend(['Vertices','Features'])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    ls1, ls_ind1 = asteroid.pointfind(np.array([1,-1,0]), asteroid.verts)
    ls2, ls_ind2 = asteroid.pointfind(np.array([0, 1, 0]), asteroid.verts)
    print(ls1)
    print(ls_ind1)
    print(ls2)
    print(ls_ind2)
    plt.show()
    print('------')
    print(asteroid.verts[0:5])
    print(asteroid.facets[0:5, 0:3])
    print(asteroid.facet_centroids[0:5])
    print(asteroid.verts[asteroid.facets[0:5, 0:3]])
    print(asteroid.facet_normals)
    #help(Asteroid)
    #asteroid.plotpoly()
    #plt.show()
    #print('x')
    #print(np.max(asteroid.verts[:, 0]) - np.min(asteroid.verts[:, 0]))
    #print('y')
    #print(np.max(asteroid.verts[:, 1]) - np.min(asteroid.verts[:, 1]))
    #print('z')
    #print(np.max(asteroid.verts[:, 2]) - np.min(asteroid.verts[:, 2]))
    #print(asteroid.rho)
    #print(asteroid.polyhedral_volume())



if __name__ == '__main__':
    main()