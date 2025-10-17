from .teacher import load_data as teacher_load_data
from .teacher_3d import load_data as teacher_3d_load_data

dataloaders = {
	"teacher"		: teacher_load_data ,
	"teacher_3d"	: teacher_3d_load_data , 
}