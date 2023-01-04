import cv2
import dlib
import numpy as np
import tqdm

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

#read the source face image and convert it to grayscale
source_image = cv2.imread("a.png")
source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

#get the presaved video stream
file_video_stream = cv2.VideoCapture("jonbe.mp4")
length = int(file_video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#print( length )

frame_width = int(file_video_stream.get(3))
frame_height = int(file_video_stream.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('singer.avi', cv2.VideoWriter_fourcc(*'MJPG'),24, size)

#loop through the frames
for _ in tqdm.trange (length):
    #get the frame from file_video_stream
    ret, current_frame = file_video_stream.read()
    #read the destination face image and convert it to grayscale
    destination_image = current_frame
    if destination_image is not None:
        destination_image_grayscale = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
        
        #create a zeros array canvas exactly like the same size of source_image_grayscale
        source_image_canvas = np.zeros_like(source_image_grayscale)
        
        #getting the shape of destination_image
        height, width, no_of_channels = destination_image.shape
        
        #create a zeros array canvas like the destination image
        destination_image_canvas = np.zeros((height,width,no_of_channels),np.uint8)
        
        #FOR THE SOURCE IMAGE
        ####################
        #Find the faces in source image
        #Returns a numpy array containing a histogram of pixels in the image
        source_faces = frontal_face_detector(source_image_grayscale)
        
        #loop through all faces found in the source image
        for source_face in source_faces:
            #predictor takes human face as input and returns the list of facial landmarks
            source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
            source_face_landmark_points = []
            
            #loop through all the 81 landmark points 
            #add them into a tuple
            for landmark_no in range(0,81):
                x_point = source_face_landmarks.part(landmark_no).x
                y_point = source_face_landmarks.part(landmark_no).y
                source_face_landmark_points.append((x_point, y_point))
      
            #converting the points into a numpy array
            source_face_landmark_points_array = np.array(source_face_landmark_points,np.int32)
            #find the convex hull of the face, the contour points
            source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)
        
            #draw a filled polygon over the zero array canvas of source
            cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)
          
            #place the created mask over the source image
            source_face_image = cv2.bitwise_and(source_image,source_image,mask=source_image_canvas)
  
            #Delaunay Triangulation of Source Image Steps
            #############################################
            #Drawing an approximate bounding rectangle around the face convex hull
            bounding_rectangle = cv2.boundingRect(source_face_convexhull)
        
            #create an empty Delaunay subdivision
            subdivisions = cv2.Subdiv2D(bounding_rectangle)
            #insert the face landmark points into subdivisions
            subdivisions.insert(source_face_landmark_points)
            #will return triangles list as 6 numbered vectors
            triangles_vector = subdivisions.getTriangleList()
            #convert vector into numpy array
            triangles_array = np.array(triangles_vector,dtype=np.int32)
            
            #print(triangles_array)
            source_triangle_index_points_list = []
            
            for triangle in triangles_array:
                index_point1 = (triangle[0], triangle[1])
                index_point2 = (triangle[2], triangle[3])
                index_point3 = (triangle[4], triangle[5])
                
                #convert the co-ordinates into facial landmark references
                index_point1 = np.where((source_face_landmark_points_array == index_point1).all(axis=1))[0][0]
                index_point2 = np.where((source_face_landmark_points_array == index_point2).all(axis=1))[0][0]
                index_point3 = np.where((source_face_landmark_points_array == index_point3).all(axis=1))[0][0]
                
                triangle = [index_point1, index_point2, index_point3]
                source_triangle_index_points_list.append(triangle)
                
        #print(triangle_index_points_list)
           
        #FOR THE DESTINATION IMAGE
        ##########################
        #Find the faces in destination image
        #Returns a numpy array containing a histogram of pixels in the image
        destination_faces = frontal_face_detector(destination_image_grayscale)

        #loop through all faces found in the destination image
        for destination_face in destination_faces:
            #predictor takes human face as input and returns the list of facial landmarks
            destination_face_landmarks = frontal_face_predictor(destination_image_grayscale, destination_face)
            destination_face_landmark_points = []
            
            #loop through all the 81 landmark points 
            #add them into a tuple
            for landmark_no in range(0,81):
                x_point = destination_face_landmarks.part(landmark_no).x
                y_point = destination_face_landmarks.part(landmark_no).y
                destination_face_landmark_points.append((x_point, y_point))
        
            #converting the points into a numpy array
            destination_face_landmark_points_array = np.array(destination_face_landmark_points,np.int32)
            #find the convex hull of the face, the contour points
            destination_face_convexhull = cv2.convexHull(destination_face_landmark_points_array)
        
        for triangle_index_points in (source_triangle_index_points_list):
            #for every source triangle from the list of triangles,
            #crop the bounding rectangle and extract only triangle points.
            #################################################################
            #get x and y coordinates of the vertices
            source_triangle_point1 = source_face_landmark_points[triangle_index_points[0]]
            source_triangle_point2 = source_face_landmark_points[triangle_index_points[1]]
            source_triangle_point3 = source_face_landmark_points[triangle_index_points[2]]
            #combining the three points into a numpy array
            source_triangle = np.array([source_triangle_point1,source_triangle_point2,source_triangle_point3], np.int32)
            
            #draw bounding rectangle around the triangle points and crop it for later use
            source_rectangle = cv2.boundingRect(source_triangle)
            (x,y,w,h) = source_rectangle
            cropped_source_rectangle = source_image[y:y+h, x:x+w]
            
            #remove rectangle points and keep the triangle points only for later use
            source_triangle_points = np.array([[source_triangle_point1[0] - x, source_triangle_point1[1] - y],
                                                [source_triangle_point2[0] - x, source_triangle_point2[1] - y],
                                                [source_triangle_point3[0] - x, source_triangle_point3[1]- y]], np.int32)
        
            #for every destination triangle from the list of triangles,
            #extract only triangle points and create a mask with the triangle.
            #################################################################
            #get x and y coordinates of the vertices
            destination_triangle_point1 = destination_face_landmark_points[triangle_index_points[0]]
            destination_triangle_point2 = destination_face_landmark_points[triangle_index_points[1]]
            destination_triangle_point3 = destination_face_landmark_points[triangle_index_points[2]]
            destination_triangle = np.array([destination_triangle_point1, destination_triangle_point2, destination_triangle_point3], np.int32)
            
            #Draw Bounding Rectangle around the triangle
            destination_rectangle = cv2.boundingRect(destination_triangle)
            (x, y, w, h) = destination_rectangle
            
            #crop destination rectangle and create a mask  for later use        
            cropped_destination_rectangle = source_image[h,w]
            cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)
            
            #remove the rectangle points to obtain only the triangle points for later use
            destination_triangle_points = np.array([[destination_triangle_point1[0] - x, destination_triangle_point1[1] - y],
                                [destination_triangle_point2[0] - x, destination_triangle_point2[1] - y],
                                [destination_triangle_point3[0] - x, destination_triangle_point3[1] - y]], np.int32)
        
            # triangle points over the cropped rectangle zero array mask
            cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)
        
            #warp source triangles to match the destination triangle shape
            #and place destination triangle mask over it    
            ############################################
            #converting to numpy array
            source_triangle_points = np.float32(source_triangle_points)
            destination_triangle_points = np.float32(destination_triangle_points)
            #creating the transformation matrix for warp affine method
            Matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
            #creating the warped triangle
            warped_triangle = cv2.warpAffine(cropped_source_rectangle, Matrix, (w,h))
            #placing destination rectangle mask over the warped triangle
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_destination_rectangle_mask)

            #reconstructing destination face in an empty canvas the size of destination image
            ##################################################################################  
            #steps to cut off the white lines in the triangle using a mask
            #small rectangular slice of destination canvas in the shape of warped rectange
            new_dest_face_canvas_area = destination_image_canvas[y: y+h, x: x+w]
            #convert the new small canvas to grayscale
            new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)
            #creating a mask to cut the pixels inside triangle excluding the white lines
            _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            #placing the mask created
            wraped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask = mask_created_triangle)
            #place the masked triangle inside the small canvas area
            new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, wraped_triangle)
            #place the new small canvas with triangle in it to the large destination canvas
            #at the designated location
            destination_image_canvas[y: y+h, x: x+w] = new_dest_face_canvas_area
            
        #Swap by Masking the Destination face and placing the newly created face
        #######################################################################
        # create a new canvas for final image in exactly the same size of destination image
        final_destination_canvas = np.zeros_like(destination_image_grayscale)
        
        #create the destination face mask
        final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destination_face_convexhull, 255)   
            
        #invert the face mask color
        final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)    
            
        #mask destination face
        destination_face_masked = cv2.bitwise_and(destination_image, destination_image, mask=final_destination_canvas)  
        
        #place new face into destination image
        destination_with_face = cv2.add(destination_face_masked,destination_image_canvas)       
                
        #Do seamless clone to make the attachment blend with the sorrounding pixels
        ###########################################################################   
        #finding the center point of the destination covex hull
        (x,y,w,h) = cv2.boundingRect(destination_face_convexhull)  
        destination_face_center_point = ((x+x+w)//2, (y+y+h)//2)
        
        #do the seamless clone   
        seamlesscloned_face = cv2.seamlessClone(destination_with_face, destination_image, final_destination_face_mask, destination_face_center_point, cv2.NORMAL_CLONE)
        result.write(seamlesscloned_face)
    if not ret:
      break

file_video_stream.release()