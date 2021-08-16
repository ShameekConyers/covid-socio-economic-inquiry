import pandas as pd
import requests
import io
import os
import json
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pathlib
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"


def fetch_json_map():
	if not os.path.exists("./data/states.json"):
		resp = requests.get("https://gist.githubusercontent.com/mshafrir/2646763/raw/8b0dbb93521f5d6889502305335104218454c2bf/states_hash.json")
		res = json.load(io.BytesIO(resp.content))
		f = open("./data/states.json", "w")
		json.dump(res, f)
		f.close()
	else:
		f = open("./data/states.json")
		res = json.load(f)
		f.close()

	result = {v: k for k, v in res.items()}
	return result


def fetch_and_clean_tables_from_wikipedia():
	"""
		Grabs the tables of interest from wikipedia

		Returns:
			A DF that contains macro level data for each state
	"""
	gini_url = "https://en.wikipedia.org/wiki/List_of_U.S._states_by_Gini_coefficient"
	pov_url = "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_poverty_rate"
	urb_url = "https://en.wikipedia.org/wiki/Urbanization_in_the_United_States"
	climate_url = "" ####

	urb_state_mapping = lambda x: x[:x.find('[')]

	#First we grab the dirty tables

	gini = pd.read_html(gini_url)
	gini = gini[2]  # this gets correct table from wikipedia page

	pov = pd.read_html(pov_url)
	pov = pov[2]

	urb = pd.read_html(urb_url)
	urb = urb[-1]
	urb = urb.droplevel(level= 0, axis = 1) #clean the unecessary multindex

	# climate = pd.read_html(climate_url) #TODO
	# data sourcing of climate not straightforward like others

	#Then we clean the tables such that the output is directly usable

	gini.columns = gini.columns.str.replace(' ', '_')
	pov.columns = pov.columns.str.replace(' ', '_')
	urb.columns = urb.columns.str.replace(' ', '_')


	gini = gini.rename(columns={
		'State_or_federal_district': 'state',
		'Gini_Coefficient': 'gini_coef'
	})
	gini.drop(['Rank'], axis=1, inplace=True)
	gini.set_index('state', inplace=True)
	gini.columns = gini.columns.str.lower()

	pov = pov.rename(columns={
		'State': 'state',
		'2019_Poverty_rate(percent_of_persons_in_poverty)[note_2][7]': 'pov_2019',
		'2014_Poverty_Rates_(includes_unrelated_children)': 'pov_2014'
		})
	pov.drop(['Rank', 'Supplemental_Poverty_Measure_(2017–2019_average)_(Geographically_Adjusted)'], axis=1, inplace=True)
	pov.set_index('state', inplace = True)
	pov.columns = pov.columns.str.lower()


	urb = urb.rename(columns={'State/Territory': 'state',
		'2010': 'urb_2010',
		'2000': 'urb_2000' })
	urb = urb[['state', 'urb_2010', 'urb_2000']].copy()
	urb['state'] = urb['state'].apply(urb_state_mapping)
	urb.set_index('state', inplace=True)
	urb.columns = urb.columns.str.lower()

	#join them all
	macro_df = gini.merge(pov, 'inner', 'state').merge(urb, 'inner', 'state')
	return macro_df.dropna()

def fetch_mental_health_df():
	"""
		Performs a sync get request to grab the data
	"""
	res = requests.get('https://data.cdc.gov/api/views/yni7-er2q/rows.csv?accessType=DOWNLOAD')
	mental_health = pd.read_csv(io.BytesIO(res.content))
	return mental_health


def chop_and_clean_mental_health(by_group: str, subgroup_name, mental_health):
	"""

		Chops the original df by_group and return a multindex of [time, subgroup_name

		Returns:
			chopped & cleaned df with a multi-index of [time, subgroup_name]
	"""

	to_drop= ['Group',
		'Time Period Label',
		'Time Period End Date',
		'Suppression Flag',
		'State',
		'Time Period',
		'index',
		'Quartile Range',
		'Confidence Interval']
	indics_mapping = {
		'Needed Counseling or Therapy But Did Not Get It, Last 4 Weeks':
		'none',

		'Received Counseling or Therapy, Last 4 Weeks':
		'therapy',

		'Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy, Last 4 Weeks':
		'either',

		'Took Prescription Medication for Mental Health, Last 4 Weeks':
		'medication'
	}
	education_mapping = {
		"Some college/Associate's degree":
		'associates',

		"Bachelor's degree or higher":
		'college',

		'High school diploma or GED':
		'highschool',

		'Less than a high school diploma':
		'none'
	}
	state_mapping = fetch_json_map()

	indics_to_col_map = {}


	df = mental_health[mental_health['Group'] == '{}'.format(by_group)].copy()
	df.reset_index(inplace=True)
	df.drop(to_drop, axis =1, inplace=True)
	df.columns = df.columns.str.replace(' ', '_')
	df.columns = df.columns.str.lower()
	df = df.rename(columns = {
		'time_period_start_date': 'time_period',
		'subgroup': '{}'.format(subgroup_name)
		})

	if subgroup_name == 'education':
		df['education'] = df['education'].apply(lambda x: education_mapping[x])
	elif subgroup_name == 'state':
		macro_df = fetch_and_clean_tables_from_wikipedia()
		df = df.merge(macro_df, 'inner', 'state')
		df['state'] = df['state'].apply(lambda x: state_mapping[x])


	df.set_index(['time_period', '{}'.format(subgroup_name)], inplace=True)

	#Here we map the indicator varibles to a summarized form.
	#A set was used to get all unique values in the columns
	df['indicator'] = df['indicator'].apply(lambda x: indics_mapping[x])

	return df

def fetch_data():
	if not (os.path.exists('./data/state.h5') and os.path.exists('./data/education.h5')):
		mental_health = fetch_mental_health_df()
		state_df = chop_and_clean_mental_health("By State", 'state', mental_health)
		education_df = chop_and_clean_mental_health(
			"By Education", 'education', mental_health)
		state_df.to_hdf('./data/state.h5', key = 'df', mode = 'w')
		education_df.to_hdf('./data/education.h5', key = 'df', mode = 'w')

	else:
		state_df = pd.read_hdf('./data/state.h5', key = 'df')
		education_df = pd.read_hdf('./data/education.h5', key = 'df')

	return [state_df, education_df]

def main():

	[state_df, education_df] = fetch_data()
	print(state_df, education_df)

	#First we get begin and end slices
	#we are only concerned with those who needed mental health care but got none
	beg = state_df.loc['08/19/2020']
	end = state_df.loc['06/23/2021']

	beg = beg[beg['indicator'] == 'none']
	end = end[end['indicator'] == 'none']

	beg = beg.rename(columns = {'value': 'unmet_mental_health'})
	end = end.rename(columns = {'value': 'unmet_mental_health'})

	#Next we test to see which states had a statistically significant drop
	#and ones that had a statistically significant increase
	low_test_mask = end['unmet_mental_health'] < beg['lowci']
	high_test_mask = end['unmet_mental_health'] > beg['highci']

	lows = low_test_mask * 1
	highs = high_test_mask * 1
	end['improved_unmet_mental_health'] = low_test_mask
	end['worsened_unmet_mental_health'] = high_test_mask

	#make a figure for our findings
	fig = px.choropleth(end,
	locations = end.index,
	color = 'improved_unmet_mental_health',
	hover_name = end.index,
	locationmode= 'USA-states')

	fig.update_layout(
		title_text = 'U.S. States',
		geo_scope = 'usa'
	)
	fig.write_image("./figures/improved.png")

	fig = px.choropleth(end,
	locations = end.index,
	color = 'worsened_unmet_mental_health',
	hover_name = end.index,
	locationmode= 'USA-states')

	fig.update_layout(
		title_text = 'U.S. States',
		geo_scope = 'usa'
	)
	fig.write_image("./figures/worsened.png")


	#Then we do sample correations.
	#note we have to strip the percentages
	print(end.columns)
	subj_df = pd.DataFrame([
		end['unmet_mental_health'],
		end['gini_coef'],
		end['pov_2019'].str.rstrip('%').astype('float'),
		end['urb_2010'].str.rstrip('%').astype('float'),
		]
	)
	subj_df = subj_df.transpose()
	corr_df = subj_df.corr()

	#remove duplicated info
	mask = np.tril(np.ones(corr_df.shape)).astype(np.bool)
	lt_corr_df = corr_df.where(mask).round(2)
	print(lt_corr_df)

	fig = ff.create_annotated_heatmap(
		z = lt_corr_df.to_numpy(),
		x = lt_corr_df.columns.tolist(),
		y = lt_corr_df.index.tolist(),
		zmax = 1,
		zmin = -1,
    colorscale = px.colors.diverging.RdBu

	)

	#We output the correlations to a figure

	fig.update_layout(
    yaxis_autorange='reversed',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
		uniformtext_minsize= 16
	)
	fig.write_image("./figures/corrs.png")

	# We do tests of multicolinearity (if we need to )

	#Now we use a simple multilinear regression on the data, only to test for
	# statistical significance
	dep_vars = end[['gini_coef']]
	dep_vars['urb_2010'] = end['urb_2010'].str.rstrip('%').astype('float')
	dep_vars['pov_2019'] = end['pov_2019'].str.rstrip('%').astype('float')
	y_var = end['unmet_mental_health']
	dep_vars = sm.add_constant(dep_vars)
	est = sm.OLS(y_var.astype(float), dep_vars.astype(float), missing='drop').fit()

	plt.rc(
		'figure',
		figsize=(12, 7))
	plt.text(
		0.01,
		0.05,
		str(est.summary()),
		{'fontsize': 10},
		fontproperties = 'monospace')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig('./figures/model0.png')

	dep_vars = end['urb_2010'].str.rstrip('%').astype('float')
	y_var = end['unmet_mental_health']
	dep_vars = sm.add_constant(dep_vars)
	est = sm.OLS(y_var.astype(float), dep_vars.astype(float), missing='drop').fit()

	plt.clf()
	dep_vars = end[['gini_coef']]
	dep_vars['pov_2019'] = end['pov_2019'].str.rstrip('%').astype('float')
	y_var = end['unmet_mental_health']
	dep_vars = sm.add_constant(dep_vars)
	est = sm.OLS(y_var.astype(float), dep_vars.astype(float), missing='drop').fit()

	plt.rc(
		'figure',
		figsize=(12, 7))
	plt.text(
		0.01,
		0.05,
		str(est.summary()),
		{'fontsize': 10},
		fontproperties = 'monospace')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig('./figures/model1.png')

	dep_vars = end['urb_2010'].str.rstrip('%').astype('float')
	y_var = end['unmet_mental_health']
	dep_vars = sm.add_constant(dep_vars)
	est = sm.OLS(y_var.astype(float), dep_vars.astype(float), missing='drop').fit()

	plt.clf()
	plt.rc(
		'figure',
		figsize=(12, 7))
	plt.text(
		0.01,
		0.05,
		str(est.summary()),
		{'fontsize': 10},
		fontproperties = 'monospace')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig('./figures/model2.png')









if __name__ == '__main__':
	os.chdir(pathlib.Path(__file__).parent.parent.resolve())
	if not os.path.exists('./data/'):
		os.mkdir('./data/')
	main()
